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
import hashlib
import torch
import lmdb
from pathlib import Path
from typing import Any, Dict, List, Optional
from torch.utils.data import Dataset
import scanpy as sc
from torch.nn.utils.rnn import pad_sequence

from .cluster_feature_loader import ClusterFeatureLoader


# ============================================================
# 模块级多进程 worker（用于 SFTDataset._load_cluster_data）
# ============================================================
_sft_worker_target_id_set = None
_sft_worker_key_cache_dir = None


def _sft_cache_name_for_db(db_path: Path) -> str:
    """用完整路径的 md5 前缀避免不同目录下同名 DB 的 key_cache 冲突。"""
    full_str = str(db_path.resolve())
    tag = hashlib.md5(full_str.encode('utf-8')).hexdigest()[:16]
    return f"{db_path.name}.{tag}"


def _sft_init_worker(target_id_set, key_cache_dir):
    """Pool initializer: 将 target_id_set / key_cache_dir 写入全局变量供 worker 复用。"""
    global _sft_worker_target_id_set, _sft_worker_key_cache_dir
    _sft_worker_target_id_set = target_id_set
    _sft_worker_key_cache_dir = key_cache_dir


def _sft_scan_db_worker(db_path_str: str):
    """扫描单个 DB，优先复用 key_cache，否则回退到 LMDB cursor。返回 (cell_id_to_loc_dict, filtered_count)。"""
    from pathlib import Path
    import lmdb
    import numpy as np

    db_path = Path(db_path_str)
    db_name = db_path.name
    local_map = {}
    filtered = 0
    target_id_set = _sft_worker_target_id_set
    key_cache_dir = _sft_worker_key_cache_dir

    # 1) 优先尝试 key_cache（已预先生成，纯文件顺序读，跳过 LMDB 初始化）
    if key_cache_dir is not None:
        cache_name = _sft_cache_name_for_db(db_path)
        bin_path = Path(key_cache_dir) / f"{cache_name}.keys.bin"
        idx_path = Path(key_cache_dir) / f"{cache_name}.keys.idx.npy"
        if bin_path.exists() and idx_path.exists():
            try:
                offsets = np.load(str(idx_path), mmap_mode='r')
                n_keys = len(offsets) - 1
                with open(bin_path, 'rb') as fh:
                    for i in range(n_keys):
                        length = int(offsets[i + 1] - offsets[i])
                        key_bytes = fh.read(length)
                        cell_id = key_bytes.decode("utf-8") if hasattr(key_bytes, "decode") else str(key_bytes)
                        if target_id_set is not None and len(target_id_set) > 0:
                            if cell_id not in target_id_set:
                                filtered += 1
                                continue
                        local_map[cell_id] = (db_name, bytes(key_bytes))
                return local_map, filtered
            except Exception:
                # key_cache 损坏或不兼容，回退到 LMDB
                pass

    # 2) 回退到 LMDB cursor
    env = lmdb.Environment(str(db_path), readonly=True, lock=False, readahead=False, meminit=False)
    with env.begin(write=False) as txn:
        for key, _ in txn.cursor():
            if key.startswith(b'-'):
                continue
            cell_id = key.decode("utf-8") if hasattr(key, "decode") else str(key)
            if target_id_set is not None and len(target_id_set) > 0:
                if cell_id not in target_id_set:
                    filtered += 1
                    continue
            local_map[cell_id] = (db_name, bytes(key))
    env.close()
    return local_map, filtered


class SFTDataset(Dataset):
    """
    SFT 多模态对齐数据集 - Map-style，读取 cluster LMDB。
    """

    # 类级缓存：所有实例共享 cell_id_to_loc（cluster data 与 curriculum 无关）
    _shared_cell_id_to_loc = None

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

        # Support multiple SFT LMDB directories; fall back to single cluster_lmdb_dir for backward compat
        _sft_dirs = self.data_config.get('sft_cluster_lmdb_dirs', [])
        if _sft_dirs:
            self.cluster_lmdb_dirs = [Path(d) for d in (_sft_dirs if isinstance(_sft_dirs, list) else [_sft_dirs])]
        elif cluster_lmdb_dir:
            self.cluster_lmdb_dirs = [Path(cluster_lmdb_dir)]
        else:
            self.cluster_lmdb_dirs = [Path(self.data_config.get('cluster_lmdb_dir', ''))]

        self.codebook_dir = Path(codebook_dir) if codebook_dir else Path(self.data_config.get('codebook_dir', ''))
        self.scgpt_gene_vocab = scgpt_gene_vocab if scgpt_gene_vocab else self.data_config.get('scgpt_gene_vocab', '')

        # Dedicated SFT key_cache dir; fall back to key_cache_dir/sft
        _sft_kc = self.data_config.get('sft_key_cache_dir', '')
        if _sft_kc:
            self.key_cache_dir = Path(_sft_kc)
        else:
            _kc = self.data_config.get('key_cache_dir', '')
            self.key_cache_dir = Path(_kc) / 'sft' if _kc else None
        if self.key_cache_dir:
            self.key_cache_dir.mkdir(parents=True, exist_ok=True)

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

        self._build_celltype_map_from_h5ad()

        # Build valid indices aligned with curriculum
        self.valid_indices = None
        if self.curriculum_stage is not None:
            self._build_valid_indices(is_main)

    @staticmethod
    def _stable_celltype_id(celltype_name: str) -> int:
        name = (celltype_name or "").strip().lower()
        if not name:
            return 0
        h = hashlib.blake2b(name.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(h, byteorder="big", signed=False) & ((1 << 63) - 1)

    @staticmethod
    def _resolve_celltype_column(obs_columns: List[str], preferred: Optional[str] = None) -> Optional[str]:
        cols = set(str(c) for c in obs_columns)
        if preferred and preferred in cols:
            return preferred
        candidates = [
            'cell_type', 'cell_type_ontology_term_id', 'cluster_label', 'leiden',
            'celltype_name', 'cell_type_name', 'celltype',
            'cell_ontology_class', 'cell_ontology_term_id', 'cell_label'
        ]
        for c in candidates:
            if c in cols:
                return c
        return None

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

    def _build_celltype_map_from_h5ad(self):
        # 对齐 run/okrcell_inference_sft.py 的两数据源
        default_sources = [
            {                                                     
          "h5ad_path": "/mnt/c20250607/user/wanghaoran/zyh/datasets/full_data.h5ad",                                                                                                             
          "json_path": "/mnt/c20250607/user/wanghaoran/zxy/data_and_features/per-gene-feature/sft_data_gene_feature/sft_data_gene_feature/sft_data_gene_feature/cw_sft_data/all_census_target_ids.json",                                                                                                            
          "celltype_col": "cell_type"                                                                                                                              
            },                                                                                                                                                           
            {                                                                                                                                                            
          "h5ad_path": "/mnt/c20250607/user/wanghaoran/zyh/datasets/cellxgene.h5ad",                                                                                                                
          "json_path": "/mnt/c20250607/user/wanghaoran/zxy/data_and_features/per-gene-feature/sft_data_gene_feature/sft_data_gene_feature/sft_data_gene_feature/cw_sft_data/all_archs4_target_ids.json",                                                                                                            
          "celltype_col": "cluster_label"                                                                                                                          
            }   
        ]

        sources = self.data_config.get('sft_celltype_sources', default_sources)
        is_main = (not torch.distributed.is_initialized()) or (torch.distributed.get_rank() == 0)

        import pickle
        _tmp_path = Path(self.key_cache_dir) / "_celltype_map.pkl" if self.key_cache_dir else Path("/tmp/_celltype_map.pkl")

        # 复用类级缓存：同一进程中多个 SFTDataset 实例（如 EASY + COMPLEX）共享结果
        if hasattr(SFTDataset, '_shared_celltype_map') and SFTDataset._shared_celltype_map is not None:
            self.celltype_by_cell_id = SFTDataset._shared_celltype_map
            if is_main:
                print(f"[SFTDataset-rankloss] Reusing in-memory celltype map: {len(self.celltype_by_cell_id):,} cells")
            return

        # 复用已写入的缓存文件（避免重复读取 h5ad）
        if _tmp_path.exists():
            with open(_tmp_path, 'rb') as f:
                celltype_map = pickle.load(f)
            self.celltype_by_cell_id = celltype_map
            SFTDataset._shared_celltype_map = celltype_map
            if is_main:
                print(f"[SFTDataset-rankloss] Reusing cached celltype map: {len(celltype_map):,} cells from {_tmp_path}")
            return

        try:
            if is_main:
                print(f"[SFTDataset-rankloss] Building celltype map from {len(sources)} h5ad sources...")
                celltype_map: Dict[str, str] = {}

                for src in sources:
                    h5ad_path = str(src.get('h5ad_path', ''))
                    json_path = str(src.get('json_path', ''))
                    preferred_col = str(src.get('celltype_col', '') or '')

                    print(f"  → try h5ad_path={h5ad_path}")
                    print(f"    json_path={json_path}, preferred_col={preferred_col}")

                    if not h5ad_path or not Path(h5ad_path).exists():
                        print(f"    ⚠️ h5ad file NOT FOUND, skip")
                        continue

                    target_ids = self._load_target_ids(json_path) if json_path else []
                    print(f"    loaded {len(target_ids):,} target_ids from json")
                    target_set = set(target_ids) if target_ids else None

                    try:
                        ad = sc.read_h5ad(h5ad_path, backed='r')
                        print(f"    h5ad loaded: n_obs={ad.n_obs:,}, n_vars={ad.n_vars:,}")
                        print(f"    obs columns: {list(ad.obs.columns)}")
                    except Exception as e:
                        print(f"    ⚠️ h5ad read failed: {e}")
                        continue

                    col = self._resolve_celltype_column(list(ad.obs.columns), preferred=preferred_col)
                    if col is None:
                        print(f"    ⚠️ celltype column not found (tried '{preferred_col}'), skip")
                        try:
                            ad.file.close()
                        except Exception:
                            pass
                        continue
                    print(f"    using celltype column: '{col}'")

                    obs_names = [str(x) for x in ad.obs_names]
                    if target_set is not None and len(target_set) > 0:
                        matched_ids = [cid for cid in obs_names if cid in target_set]
                        print(f"    target_set size={len(target_set):,}, matched_ids={len(matched_ids):,}")
                    else:
                        matched_ids = obs_names
                        print(f"    no target filter, all obs_names={len(matched_ids):,}")

                    if len(matched_ids) == 0:
                        print(f"    ⚠️ matched_ids empty, skip")
                        try:
                            ad.file.close()
                        except Exception:
                            pass
                        continue

                    try:
                        values = ad.obs.loc[matched_ids, col].astype(str).tolist()
                        print(f"    extracted {len(values):,} celltype values")
                    except Exception as e:
                        print(f"    ⚠️ extract celltype values failed: {e}, fallback to empty")
                        values = [""] * len(matched_ids)

                    for cid, val in zip(matched_ids, values):
                        if cid not in celltype_map:
                            celltype_map[cid] = str(val or "")

                    try:
                        ad.file.close()
                    except Exception:
                        pass

                    print(f"    ✅ source={Path(h5ad_path).name}, col={col}, mapped={len(matched_ids):,}")

                with open(_tmp_path, 'wb') as f:
                    pickle.dump(celltype_map, f)
                print(f"[SFTDataset-rankloss] celltype map ready: {len(celltype_map):,} cells, saved to {_tmp_path}")
        finally:
            if torch.distributed.is_initialized():
                if is_main:
                    print(f"[SFTDataset-rankloss] Rank {torch.distributed.get_rank()} entering celltype barrier...")
                torch.distributed.barrier()
                if is_main:
                    print(f"[SFTDataset-rankloss] Rank {torch.distributed.get_rank()} passed celltype barrier.")

        with open(_tmp_path, 'rb') as f:
            self.celltype_by_cell_id = pickle.load(f)

    def _lookup_celltype_name(self, cell_id_str: str) -> str:
        return str(self.celltype_by_cell_id.get(str(cell_id_str), "") or "")

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

    def _build_sft_target_id_set(self):
        paths = self.data_config.get('sft_target_id_json_paths', [])
        if isinstance(paths, str):
            paths = [paths]
        target_ids = set()
        for jp in paths:
            if not jp:
                continue
            p = Path(str(jp))
            if not p.exists():
                continue
            ids = self._load_target_ids(str(p))
            target_ids.update(ids)
        return target_ids

    def _build_sft_key_cache_if_needed(self, db_path: Path):
        """为单个 DB 生成 key_cache（如果不存在）。所有 rank 各自为自己分配的 DB 生成，无竞争。"""
        if self.key_cache_dir is None:
            return
        cache_name = _sft_cache_name_for_db(db_path)
        bin_path = self.key_cache_dir / f"{cache_name}.keys.bin"
        idx_path = self.key_cache_dir / f"{cache_name}.keys.idx.npy"
        if bin_path.exists() and idx_path.exists():
            return

        import numpy as np
        env = lmdb.Environment(str(db_path), readonly=True, lock=False, readahead=False, meminit=False)
        offsets = [0]
        cur = 0
        with env.begin(write=False) as txn, open(bin_path, 'wb') as f:
            for k, _ in txn.cursor():
                if k.startswith(b'-'):
                    continue
                b = bytes(k)
                f.write(b)
                cur += len(b)
                offsets.append(cur)
        env.close()
        arr = np.asarray(offsets, dtype=np.uint64)
        np.save(idx_path, arr)

    def _load_cluster_data(self, is_main: bool):
        # 优先复用类级缓存（EASY 构建后 COMPLEX 直接复用，cluster data 与 curriculum 无关）
        if SFTDataset._shared_cell_id_to_loc is not None:
            self.cell_id_to_loc = SFTDataset._shared_cell_id_to_loc.copy()
            if is_main:
                print(f"[SFTDataset] Reusing cached cell_id_to_loc: {len(self.cell_id_to_loc):,} cells")
            return

        # 从多个目录收集 DB 文件，支持可配置 glob 模式
        glob_pattern = self.data_config.get('sft_db_glob_pattern', '*.db')
        cluster_db_files = []
        for lmdb_dir in self.cluster_lmdb_dirs:
            if lmdb_dir.exists():
                cluster_db_files.extend(lmdb_dir.glob(glob_pattern))
        cluster_db_files = sorted(set(cluster_db_files))
        if len(cluster_db_files) == 0:
            dirs_str = ', '.join(str(d) for d in self.cluster_lmdb_dirs)
            raise ValueError(f"No DB files matching '{glob_pattern}' found in [{dirs_str}]")

        if is_main:
            print(f"[SFTDataset] Total {len(cluster_db_files)} DBs found")
            print(f"[SFTDataset] key_cache_dir={self.key_cache_dir}")

        # key_cache 生成 + 扫描：只有 rank 0 执行，其他 rank 等待后读取共享缓存文件
        import pickle
        _tmp_path = Path(self.key_cache_dir) / "_shared_cell_id_to_loc.pkl" if self.key_cache_dir else Path("/tmp/_shared_cell_id_to_loc.pkl")

        try:
            if self.rank == 0:
                # 1) 生成缺失的 key_cache
                for db_path in cluster_db_files:
                    self._build_sft_key_cache_if_needed(db_path)

                # 2) rank 0 单进程扫描全部 DB（避免 multiprocessing Pool 在 NCCL 环境中死锁）
                target_id_set = self.sft_target_id_set
                cell_id_to_loc = {}
                filtered_out = 0
                _kc = str(self.key_cache_dir) if self.key_cache_dir else None
                _sft_init_worker(target_id_set, _kc)
                for db_path in cluster_db_files:
                    local_map, filtered = _sft_scan_db_worker(str(db_path))
                    cell_id_to_loc.update(local_map)
                    filtered_out += filtered

                # 3) 写入共享缓存文件
                with open(_tmp_path, 'wb') as f:
                    pickle.dump(cell_id_to_loc, f)
                if is_main:
                    print(f"[SFTDataset] Rank 0 indexed {len(cell_id_to_loc):,} cells, saved to {_tmp_path}")
                    if self.sft_use_target_id_filter:
                        n_target = len(self.sft_target_id_set) if self.sft_target_id_set is not None else 0
                        print(f"[SFTDataset] target-id filter enabled: target_ids={n_target:,}, filtered_out={filtered_out:,}")
        finally:
            # 4) 所有 rank 同步：无论 rank 0 成功/失败/异常，都必须走到 barrier，否则其他 rank 会永远阻塞
            if torch.distributed.is_initialized():
                if is_main:
                    print(f"[SFTDataset] Rank {self.rank} entering barrier...")
                torch.distributed.barrier()
                if is_main:
                    print(f"[SFTDataset] Rank {self.rank} passed barrier.")

        with open(_tmp_path, 'rb') as f:
            cell_id_to_loc = pickle.load(f)

        self.cell_id_to_loc = cell_id_to_loc
        SFTDataset._shared_cell_id_to_loc = cell_id_to_loc  # 缓存到类变量

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

            # 按 rank 分片，每张卡只保留 1/world_size 的样本
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

        celltype_name = self._lookup_celltype_name(cell_id_str)
        celltype_id = self._stable_celltype_id(celltype_name)

        out = {
            'input_ids': input_ids_tensor,
            'position_ids': position_ids_tensor,
            'attention_mask': torch.ones_like(input_ids_tensor, dtype=torch.bool),
            'labels': text_labels_tensor,
            'modality_positions': torch.tensor([[gene_start_pos, num_gene_tokens]], dtype=torch.long),
            'gene_mask': (input_ids_tensor[gene_start_pos:gene_start_pos+num_gene_tokens] == self.mask_gene_id),
            'non_zero_mask': non_zero_mask,
            'celltype_name': celltype_name,
            'celltype_id': torch.tensor(celltype_id, dtype=torch.long),
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
