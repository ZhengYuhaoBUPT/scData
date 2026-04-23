# coding=utf-8
"""
Cell-Only SFT Dataset for Multimodal Single-Cell Conversation Ablation.

Current input path:
- conversation JSON -> cell id
- expression H5AD -> top-1200 expressed genes for that cell
- static gene embedding checkpoint -> map genes to 768d embeddings
- Q-Former consumes [1200, 768] and compresses to pathway tokens
"""

import collections
import json
import random
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from src_ablation_cw.datasets.expression_h5ad_registry import ExpressionH5ADRegistry


class CWSFTCellOnlyDataset(Dataset):
    def __init__(
        self,
        feature_dir: Optional[str],
        json_paths: List[str],
        text_tokenizer: Any,
        config_dict: Dict[str, Any],
        special_tokens_ids: Dict[str, int],
        accelerator=None,
        curriculum_stage: str = None,
        data_type_tag: str = "stage2",
        append_image_tag: bool = True,
    ):
        super().__init__()

        self.dataset_config = config_dict.get("dataset", {})
        self.max_seq_len = int(self.dataset_config.get("max_seq_len", 1024))
        self.cell_feature_tokens = int(self.dataset_config.get("cell_feature_tokens", 8))
        self.cell_feature_dim = int(self.dataset_config.get("cell_feature_dim", 768))
        self.gene_input_tokens = int(self.dataset_config.get("gene_input_tokens", 1200))
        self.gene_rank_order = str(self.dataset_config.get("gene_rank_order", "desc"))

        self.data_type_tag = data_type_tag
        self.append_image_tag = append_image_tag
        self.curriculum_stage = curriculum_stage

        data_cfg = config_dict.get("data", {})
        model_cfg = config_dict.get("model", {})
        self.gene_h5ad_paths = data_cfg.get("gene_h5ad_paths", [])
        static_gene_ckpt_path = model_cfg.get("static_gene_embedding_ckpt_path")
        if not self.gene_h5ad_paths:
            raise ValueError("config.data.gene_h5ad_paths is required")
        if not static_gene_ckpt_path:
            raise ValueError("model.static_gene_embedding_ckpt_path is required for gene-token training")

        self.registry = ExpressionH5ADRegistry(
            h5ad_paths=self.gene_h5ad_paths,
            static_gene_ckpt_path=static_gene_ckpt_path,
            gene_input_tokens=self.gene_input_tokens,
            gene_rank_order=self.gene_rank_order,
        )

        self.accelerator = accelerator
        self.num_replicas = accelerator.num_processes if accelerator else 1
        self.rank = accelerator.process_index if accelerator else 0

        self.text_tokenizer = text_tokenizer
        self.soc_id = special_tokens_ids.get('soc_id')
        self.eoc_id = special_tokens_ids.get('eoc_id')

        self.bos_id = getattr(text_tokenizer, 'bos_token_id', None)
        self.pad_id = text_tokenizer.pad_token_id if getattr(text_tokenizer, 'pad_token_id', None) is not None else 151643

        is_main = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0

        self.qa_map = {}
        for jp in json_paths:
            if is_main:
                print(f"📥 加载对话文件: {jp}")
            with open(jp, 'r') as f:
                data = json.load(f)
                for item in data:
                    cell_id = str(item['id'])
                    self.qa_map.setdefault(cell_id, []).append(item['conversations'])
        if is_main:
            print(f"✅ Unique cell ids in QA map: {len(self.qa_map):,}")

        all_cell_ids_raw = sorted(self.qa_map.keys())
        all_cell_ids = [cid for cid in all_cell_ids_raw if self.registry.has_cell(cid)]
        dropped_cell_ids = len(all_cell_ids_raw) - len(all_cell_ids)
        if is_main and dropped_cell_ids > 0:
            print(f"[CWSFTCellOnlyDataset] dropped {dropped_cell_ids:,} cell ids not found in gene_h5ad_paths")
        total_cells_all_files = len(all_cell_ids)
        usable_samples = (total_cells_all_files // self.num_replicas) * self.num_replicas
        cells_per_rank = usable_samples // self.num_replicas if self.num_replicas > 0 else total_cells_all_files
        my_start = self.rank * cells_per_rank
        my_end = my_start + cells_per_rank

        local_cell_ids = all_cell_ids[my_start:my_end]
        self.data_blocks = [{"cell_ids": local_cell_ids}]
        self.cumulative_sizes = [0, len(local_cell_ids)]
        self.total_cells = len(local_cell_ids)

        self.valid_indices = []
        self._build_valid_indices(is_main)

    def _build_valid_indices(self, is_main: bool) -> None:
        all_valid_indices = []
        for block_idx, block in enumerate(self.data_blocks):
            for local_idx, cell_id in enumerate(block['cell_ids']):
                conv_count = len(self.qa_map.get(str(cell_id), []))
                conv_count = max(1, conv_count)
                for conv_idx in range(conv_count):
                    all_valid_indices.append((block_idx, local_idx, conv_idx))

        local_count = len(all_valid_indices)

        if dist.is_initialized():
            world_size = dist.get_world_size()
            rank = dist.get_rank()
            count_tensor = torch.tensor(local_count, device='cuda' if torch.cuda.is_available() else 'cpu', dtype=torch.long)
            all_counts = [torch.zeros_like(count_tensor) for _ in range(world_size)]
            dist.all_gather(all_counts, count_tensor)
            counts_list = [int(c.item()) for c in all_counts]

            max_count = max(counts_list) if counts_list else local_count
            if local_count < max_count and local_count > 0:
                rnd = random.Random(42 + rank)
                pad_indices = [all_valid_indices[rnd.randrange(local_count)] for _ in range(max_count - local_count)]
                all_valid_indices.extend(pad_indices)

            self.valid_indices = all_valid_indices
            if is_main:
                print(f"📊 各卡分布: {counts_list} -> Padding 至 {max_count}")
        else:
            self.valid_indices = all_valid_indices

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        block_idx, local_idx, conv_choice = self.valid_indices[idx]
        block = self.data_blocks[block_idx]

        cell_id = block['cell_ids'][local_idx]
        cell_feature = self.registry.get_gene_tokens(str(cell_id))

        cell_id_str = str(cell_id)
        conv_bank = self.qa_map.get(cell_id_str, [])
        conversations = conv_bank[conv_choice % len(conv_bank)] if conv_bank else []

        input_ids = []
        text_labels = []

        if self.bos_id is not None:
            input_ids.append(self.bos_id)
            text_labels.append(-100)

        system_prompt = "You are a helpful assistant."
        sys_str = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        sys_tokens = self.text_tokenizer.encode(sys_str, add_special_tokens=False)
        input_ids.extend(sys_tokens)
        text_labels.extend([-100] * len(sys_tokens))

        cell_tokens = [self.soc_id] + [self.pad_id] * self.cell_feature_tokens + [self.eoc_id]
        cell_start_pos = 0
        is_first_user = True

        for conv in conversations:
            role = conv['from']
            val = conv.get('value', "")
            val = val.replace("[INST]", "").replace("[/INST]", "")
            val = val.replace("<<SYS>>", "").replace("<</SYS>>", "")
            val = val.replace("<s>", "").replace("</s>", "").strip()

            if role == 'human':
                role = 'user'
            elif role == 'gpt':
                role = 'assistant'

            if role == 'user':
                if is_first_user:
                    val = val.replace("<image>", "").strip()
                    if self.append_image_tag and val:
                        val = "\n" + val

                    u_header = self.text_tokenizer.encode("<|im_start|>user\n", add_special_tokens=False)
                    input_ids.extend(u_header)
                    text_labels.extend([-100] * len(u_header))

                    soc_pos = len(input_ids)
                    input_ids.extend(cell_tokens)
                    text_labels.extend([-100] * len(cell_tokens))
                    cell_start_pos = soc_pos + 1

                    u_body = self.text_tokenizer.encode(f"{val}<|im_end|>\n", add_special_tokens=False)
                    input_ids.extend(u_body)
                    text_labels.extend([-100] * len(u_body))

                    is_first_user = False
                else:
                    u_str = f"<|im_start|>user\n{val}<|im_end|>\n"
                    u_ids = self.text_tokenizer.encode(u_str, add_special_tokens=False)
                    input_ids.extend(u_ids)
                    text_labels.extend([-100] * len(u_ids))

            elif role == 'assistant':
                a_header = self.text_tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
                a_body = self.text_tokenizer.encode(f"{val}<|im_end|>\n", add_special_tokens=False)

                input_ids.extend(a_header)
                text_labels.extend([-100] * len(a_header))
                input_ids.extend(a_body)
                text_labels.extend(a_body)

        if len(input_ids) > self.max_seq_len:
            input_ids = input_ids[:self.max_seq_len]
            text_labels = text_labels[:self.max_seq_len]

        input_ids_tensor = torch.tensor(input_ids, dtype=torch.long)
        labels_tensor = torch.tensor(text_labels, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids_tensor, dtype=torch.bool)

        return {
            "input_ids": input_ids_tensor,
            "labels": labels_tensor,
            "attention_mask": attention_mask,
            "cell_features": cell_feature,
            "cell_positions": torch.tensor([cell_start_pos, self.cell_feature_tokens], dtype=torch.long),
            "data_type": self.data_type_tag,
        }


def cw_cell_only_collate(batch):
    batched = collections.defaultdict(list)
    for data in batch:
        for k, v in data.items():
            batched[k].append(v)

    out = {}
    for k, v in batched.items():
        if k == 'data_type':
            out[k] = v
            continue
        if isinstance(v[0], torch.Tensor):
            if v[0].dim() == 1:
                if k in ('cell_features', 'cell_positions'):
                    out[k] = torch.stack(v, dim=0)
                elif k == 'labels':
                    out[k] = pad_sequence(v, batch_first=True, padding_value=-100)
                elif k == 'input_ids':
                    out[k] = pad_sequence(v, batch_first=True, padding_value=151643)
                else:
                    out[k] = pad_sequence(v, batch_first=True, padding_value=0)
            else:
                out[k] = torch.stack(v, dim=0)
    return out
