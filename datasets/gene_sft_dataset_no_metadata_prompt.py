# coding=utf-8
"""
SFT Dataset for Multimodal Single-Cell Conversation
Fully decoupled version: reads from cluster LMDB only (gene sequence + text).
"""

import collections
import json
import os
import random
import bisect
import torch
from pathlib import Path
from typing import Any, Dict, List, Optional
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from .cluster_feature_loader import ClusterFeatureLoader


class SFTDataset(Dataset):
    """
    SFT 多模态对齐数据集 - Map-style，读取 cluster LMDB。
    """

    def __init__(self,
                 cluster_lmdb_dir: str = None,
                 codebook_dir: str = None,
                 scgpt_gene_vocab: str = None,
                 json_paths: List[str] = None,
                 text_tokenizer: Any = None,
                 config_dict: Dict[str, Any] = None,
                 special_tokens_ids: Dict[str, int] = None,
                 max_seq_len: int = 2048,
                 accelerator=None,
                 curriculum_stage: str = None):
        super().__init__()

        self.data_config = config_dict.get('data', {}) if config_dict else {}
        self.dataset_config = config_dict.get('dataset', {}) if config_dict else {}
        self.max_genes = self.dataset_config.get('max_genes', 1200)
        self.max_seq_len = self.dataset_config.get('max_seq_len', max_seq_len)

        _sft_dirs = self.data_config.get('sft_cluster_lmdb_dirs', [])
        if _sft_dirs:
            self.cluster_lmdb_dirs = [Path(d) for d in (_sft_dirs if isinstance(_sft_dirs, list) else [_sft_dirs])]
        elif cluster_lmdb_dir:
            self.cluster_lmdb_dirs = [Path(cluster_lmdb_dir)]
        else:
            self.cluster_lmdb_dirs = [Path(self.data_config.get('cluster_lmdb_dir', ''))]

        self.codebook_dir = Path(codebook_dir) if codebook_dir else Path(self.data_config.get('codebook_dir', ''))
        self.scgpt_gene_vocab = scgpt_gene_vocab if scgpt_gene_vocab else self.data_config.get('scgpt_gene_vocab', '')

        self.accelerator = accelerator
        self.num_replicas = accelerator.num_processes if accelerator else 1
        self.rank = accelerator.process_index if accelerator else 0

        self.text_tokenizer = text_tokenizer
        self.sog_id = special_tokens_ids.get('sog_id')
        self.eog_id = special_tokens_ids.get('eog_id')
        self.bos_id = getattr(text_tokenizer, 'bos_token_id', None)
        self.pad_id = text_tokenizer.pad_token_id if getattr(text_tokenizer, 'pad_token_id', None) is not None else 151643
        self.text_vocab_size = getattr(text_tokenizer, 'vocab_size', 151936)

        self.curriculum_stage = curriculum_stage

        is_main = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
        if is_main:
            print("\n" + "=" * 60 + "\n🧠 初始化 SFTDataset..." + (f" [{curriculum_stage}]" if curriculum_stage else "") + "\n" + "=" * 60)

        # Build compact gene vocab
        self._build_gene_vocab()
        self.mask_gene_id = self.text_vocab_size + self.gene_vocab_size

        # Load conversations
        json_paths = json_paths if json_paths else self.data_config.get('sft_json_paths', [])
        self.qa_map = {}
        total_records = 0
        for jp in json_paths:
            if is_main:
                print(f"📥 加载对话文件: {jp}")
            with open(jp, 'r') as f:
                data = json.load(f)
                for item in data:
                    if curriculum_stage:
                        difficulty = self._classify_difficulty(item)
                        if curriculum_stage == 'EASY' and difficulty != 'EASY':
                            continue
                        elif curriculum_stage == 'COMPLEX' and difficulty not in ['MEDIUM', 'HARD']:
                            continue
                    cell_id = str(item['id'])
                    conversations = item['conversations']
                    if cell_id not in self.qa_map:
                        self.qa_map[cell_id] = []
                    self.qa_map[cell_id].append(conversations)
                    total_records += 1

        self.total_conversation_records = total_records
        if is_main:
            print(f"✅ 共加载 {total_records:,} 条对话记录")
            print(f"   - unique cell ids: {len(self.qa_map):,}")

        # Load cluster feature index via loader
        self.cell_id_to_loc = {}
        self.sft_use_target_id_filter = bool(self.data_config.get('sft_use_target_id_filter', False))
        self.sft_target_id_set = self._build_sft_target_id_set() if self.sft_use_target_id_filter else None
        self.use_cell_cls = bool(self.data_config.get('use_cell_cls', False))
        if self.use_cell_cls:
            from .cell_aware_feature_loader import CellAwareFeatureLoader
            self.cell_aware_loader = CellAwareFeatureLoader(
                cluster_lmdb_dir=self.cluster_lmdb_dirs,
                codebook_dir=self.codebook_dir,
                clusters_per_gene=64,
                prefer_compact_codebook=bool(self.data_config.get('prefer_compact_codebook', True)),
                readahead=bool(self.data_config.get('cluster_loader_readahead', False)),
                max_readers=int(self.data_config.get('cluster_loader_max_readers', 2048)),
                gene_weight=float(self.data_config.get('cell_cls_gene_weight', 0.5)),
                cell_weight=float(self.data_config.get('cell_cls_cell_weight', 0.5)),
                cell_dropout=float(self.data_config.get('cell_cls_dropout', 0.0)),
            )
            # 仅用于索引 key
            self.feature_loader = self.cell_aware_loader._base
            if is_main:
                print(f"[SFTDataset] ✅ use_cell_cls=True, gene_weight={self.data_config.get('cell_cls_gene_weight', 0.5)}, cell_dropout={self.data_config.get('cell_cls_dropout', 0.0)}")
        else:
            self.feature_loader = ClusterFeatureLoader(
                cluster_lmdb_dir=self.cluster_lmdb_dirs,
                readahead=bool(self.data_config.get("cluster_loader_readahead", False)),
                max_readers=int(self.data_config.get("cluster_loader_max_readers", 2048)),
            )
        self._load_cluster_data(is_main)

        # Build valid indices aligned with curriculum
        self.valid_indices = None
        if self.curriculum_stage is not None:
            self._build_valid_indices(is_main)

    @staticmethod
    def _load_target_ids(json_path: str) -> List[str]:
        try:
            with open(json_path, 'r') as f:
                obj = json.load(f)
            if isinstance(obj, dict) and 'target_ids' in obj:
                return [str(x) for x in obj.get('target_ids', [])]
            if isinstance(obj, list):
                return [str(x) for x in obj]
        except Exception:
            return []
        return []

    def _build_sft_target_id_set(self):
        paths = self.data_config.get('sft_target_id_json_paths', [])
        if isinstance(paths, str):
            paths = [paths]
        target_ids = set()
        for path in paths:
            for tid in self._load_target_ids(path):
                target_ids.add(str(tid))
        return target_ids

    def _build_gene_vocab(self):
        nclusters_path = self.codebook_dir / "gene_nclusters.json"
        with open(nclusters_path, "r") as f:
            nclusters = {int(k): int(v) for k, v in json.load(f).items()}
        valid_genes = sorted([gid for gid, k in nclusters.items() if k > 0])
        # 保留所有有效基因，不再截断到 max_genes；max_genes 仅控制每个细胞的输入序列长度
        self.scgpt_to_local = {gid: idx for idx, gid in enumerate(valid_genes)}
        self.local_to_scgpt = valid_genes
        self.gene_vocab_size = len(valid_genes) * 64
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            print(f"[SFTDataset] compact gene vocab: {len(valid_genes)} genes x 64 = {self.gene_vocab_size} tokens")

    def _load_cluster_data(self, is_main: bool):
        glob_pattern = self.data_config.get('sft_db_glob_pattern', '*_cluster.db')
        cluster_db_files = []
        for cluster_dir in self.cluster_lmdb_dirs:
            cluster_db_files.extend(sorted(cluster_dir.glob(glob_pattern)))
        if len(cluster_db_files) == 0:
            raise ValueError(f"No cluster DB files found in {self.cluster_lmdb_dirs}")

        if is_main:
            print(f"[SFTDataset] Indexing {len(cluster_db_files)} cluster DBs...")

        for db_path in cluster_db_files:
            db_name = db_path.name
            for key in self.feature_loader.list_keys(db_name, exclude_meta=True):
                cell_id = key.decode("utf-8") if hasattr(key, "decode") else str(key)
                if self.sft_use_target_id_filter and self.sft_target_id_set is not None and cell_id not in self.sft_target_id_set:
                    continue
                self.cell_id_to_loc[cell_id] = (db_name, key)

        if is_main:
            print(f"[SFTDataset] Indexed {len(self.cell_id_to_loc):,} cells from cluster DBs")

    def _classify_difficulty(self, item: Dict) -> str:
        conversations = item.get('conversations', [])
        if len(conversations) == 2:
            answer = conversations[1].get('value', '')
            if len(answer) < 200:
                return 'EASY'
            else:
                return 'MEDIUM'
        else:
            return 'HARD'

    def _build_valid_indices(self, is_main: bool) -> None:
        qa_keys = set(self.qa_map.keys())
        all_valid_indices = []
        for cell_id_str in qa_keys:
            if cell_id_str not in self.cell_id_to_loc:
                continue
            conv_count = len(self.qa_map.get(cell_id_str, []))
            conv_count = max(1, conv_count)
            for conv_idx in range(conv_count):
                all_valid_indices.append((cell_id_str, conv_idx))

        local_count = len(all_valid_indices)

        if torch.distributed.is_initialized():
            import torch.distributed as dist
            world_size = dist.get_world_size()
            rank = dist.get_rank()

            self.valid_indices = all_valid_indices[rank::world_size]
            local_count = len(self.valid_indices)

            count_tensor = torch.tensor(local_count, device='cuda' if torch.cuda.is_available() else 'cpu', dtype=torch.long)
            all_counts = [torch.zeros_like(count_tensor) for _ in range(world_size)]
            dist.all_gather(all_counts, count_tensor)
            counts_list = [int(c.item()) for c in all_counts]
            global_total = sum(counts_list)
            max_count = max(counts_list) if counts_list else local_count
            if local_count < max_count:
                if local_count == 0:
                    raise RuntimeError(f"[{self.curriculum_stage}] Rank {rank} has 0 samples, cannot pad to {max_count}.")
                rnd = random.Random(42 + rank)
                pad_indices = [self.valid_indices[rnd.randrange(local_count)] for _ in range(max_count - local_count)]
                self.valid_indices.extend(pad_indices)
                local_count = len(self.valid_indices)
            print(f"✅ [{self.curriculum_stage}] Rank {rank}: 本卡有效样本 {local_count:,}，全球总计 {global_total:,}")
            if is_main:
                print(f"   📊 各卡分布: {counts_list}")
        else:
            self.valid_indices = all_valid_indices
            print(f"✅ [{self.curriculum_stage}] 有效样本：{local_count:,} (非分布式)")

    def __len__(self) -> int:
        if self.valid_indices is not None:
            return len(self.valid_indices)
        return len(self.cell_id_to_loc)

    def _build_flat_gene_tokens(self, scgpt_ids, cluster_indices):
        flat_tokens = []
        for gid, cid in zip(scgpt_ids, cluster_indices):
            local_id = self.scgpt_to_local.get(int(gid))
            if local_id is not None:
                token_id = self.text_vocab_size + (local_id * 64 + int(cid))
            else:
                token_id = self.mask_gene_id
            flat_tokens.append(token_id)
        return flat_tokens

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.valid_indices is not None:
            cell_id_str, conv_choice = self.valid_indices[idx]
        else:
            cell_id_str = list(self.cell_id_to_loc.keys())[idx]
            conv_choice = None

        db_name, key = self.cell_id_to_loc[cell_id_str]
        if self.use_cell_cls:
            raw = self.cell_aware_loader.get_raw(db_name, key)
            if raw is None:
                return self.__getitem__((idx + 1) % len(self))
            record, lookup = raw
            fused_emb, token_ids_arr, valid_mask = self.cell_aware_loader._fuse(record, lookup)
            flat_gene_tokens = []
            for tid in token_ids_arr:
                if tid >= 0:
                    flat_gene_tokens.append(int(tid) + self.text_vocab_size)
                else:
                    flat_gene_tokens.append(self.mask_gene_id)
            gene_embeddings = torch.from_numpy(fused_emb).float()
            non_zero_mask = torch.from_numpy(valid_mask.astype(bool))
            scgpt_ids = record.scgpt_ids
        else:
            record = self.feature_loader.get(db_name, key)
            if record is None:
                # 缺失 key 时跳过该样本，避免训练中断
                return self.__getitem__((idx + 1) % len(self))
            scgpt_ids = record.scgpt_ids
            cluster_indices = record.cluster_indices
            flat_gene_tokens = self._build_flat_gene_tokens(scgpt_ids, cluster_indices)
            gene_embeddings = None
            non_zero_mask = torch.tensor(scgpt_ids != 0, dtype=torch.bool)

        num_gene_tokens = len(flat_gene_tokens)

        system_prompt = "You are a helpful assistant."
        conv_bank = self.qa_map.get(cell_id_str, [])
        if conv_bank:
            if conv_choice is None:
                conv_idx = int(idx) % len(conv_bank)
            else:
                conv_idx = int(conv_choice) % len(conv_bank)
            conversations = conv_bank[conv_idx]
        else:
            conversations = [
                {"from": "human", "value": "Describe this cell (with its gene sequence)."},
                {"from": "gpt", "value": "It is a biological sample."}
            ]

        input_ids = []
        text_labels = []

        if self.bos_id is not None:
            input_ids.append(self.bos_id)
            text_labels.append(-100)

        sys_tokens = self.text_tokenizer.encode(f"<|im_start|>system\n{system_prompt}<|im_end|>\n", add_special_tokens=False)
        input_ids.extend(sys_tokens)
        text_labels.extend([-100] * len(sys_tokens))

        gene_tokens = [self.sog_id] + flat_gene_tokens + [self.eog_id]
        gene_start_pos = 0
        sog_pos = 0
        is_first_user = True

        for conv in conversations:
            if conv['from'] == 'human':
                text = str(conv.get('value', ''))
                text = text.replace("[INST]", "").replace("[/INST]", "")
                text = text.replace("<<SYS>>", "").replace("<</SYS>>", "")
                text = text.replace("<s>", "").replace("</s>", "").strip()
                if is_first_user:
                    text = text.replace("<image>", "").strip()

                prefix_tokens = self.text_tokenizer.encode("<|im_start|>user\n", add_special_tokens=False)
                input_ids.extend(prefix_tokens)
                text_labels.extend([-100] * len(prefix_tokens))

                if is_first_user:
                    sog_pos = len(input_ids)
                    input_ids.extend(gene_tokens)
                    text_labels.extend([-100] * len(gene_tokens))
                    gene_start_pos = sog_pos + 1
                    is_first_user = False

                content_tokens = self.text_tokenizer.encode(f"{text}<|im_end|>\n", add_special_tokens=False)
                input_ids.extend(content_tokens)
                text_labels.extend([-100] * len(content_tokens))

            elif conv['from'] == 'gpt':
                prefix = "<|im_start|>assistant\n"
                content = f"{conv['value']}<|im_end|>\n"
                p_tokens = self.text_tokenizer.encode(prefix, add_special_tokens=False)
                c_tokens = self.text_tokenizer.encode(content, add_special_tokens=False)
                input_ids.extend(p_tokens + c_tokens)
                text_labels.extend([-100] * len(p_tokens) + c_tokens)

        if len(input_ids) > self.max_seq_len:
            input_ids = input_ids[:self.max_seq_len]
            text_labels = text_labels[:self.max_seq_len]

        input_ids_tensor = torch.tensor(input_ids, dtype=torch.long)
        text_labels_tensor = torch.tensor(text_labels, dtype=torch.long)
        seq_len = len(input_ids_tensor)
        position_ids_tensor = torch.arange(seq_len, dtype=torch.long)

        out = {
            'input_ids': input_ids_tensor,
            'position_ids': position_ids_tensor,
            'attention_mask': torch.ones_like(input_ids_tensor, dtype=torch.bool),
            'labels': text_labels_tensor,
            'modality_positions': torch.tensor([[gene_start_pos, num_gene_tokens]], dtype=torch.long),
            'gene_mask': (input_ids_tensor[gene_start_pos:gene_start_pos+num_gene_tokens] == self.mask_gene_id),
            'non_zero_mask': non_zero_mask,
            'data_type': ['stage2']
        }
        if gene_embeddings is not None:
            out['gene_embeddings'] = gene_embeddings
        return out

    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict:
        batched = collections.defaultdict(list)
        for data in batch:
            for k, v in data.items():
                batched[k].append(v)

        for k, v in batched.items():
            if k == 'data_type':
                from itertools import chain
                batched[k] = list(chain.from_iterable(v))
            elif isinstance(v[0], torch.Tensor):
                if v[0].dim() == 1:
                    if k in ('gene_mask', 'non_zero_mask'):
                        batched[k] = torch.stack(v, dim=0)
                    elif k == 'labels':
                        batched[k] = pad_sequence(v, batch_first=True, padding_value=-100)
                    elif k == 'input_ids':
                        batched[k] = pad_sequence(v, batch_first=True, padding_value=151643)
                    elif k == 'position_ids':
                        batched[k] = pad_sequence(v, batch_first=True, padding_value=0)
                    else:
                        batched[k] = pad_sequence(v, batch_first=True, padding_value=0)
                elif v[0].dim() == 0:
                    batched[k] = torch.stack(v, dim=0)
                elif v[0].dim() == 2:
                    if k == 'modality_positions':
                        batched[k] = torch.stack(v, dim=0)
                else:
                    batched[k] = torch.stack(v, dim=0)
        return dict(batched)
