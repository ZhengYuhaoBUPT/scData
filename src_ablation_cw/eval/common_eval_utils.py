#!/usr/bin/env python3
# coding=utf-8

import copy
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import anndata as ad
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from src_ablation_cw.datasets.expression_h5ad_registry import ExpressionH5ADRegistry
from src_ablation_cw.datasets.gene_token_utils import load_static_gene_bundle


def read_cell_features(adata, local_start=None, local_end=None, flatten=False):
    """
    Unified cell feature reader with consistent priority:
    X_scFM > X_pca > X
    """
    if "X_scFM" in adata.obsm:
        source = adata.obsm["X_scFM"]
    elif "X_pca" in adata.obsm:
        source = adata.obsm["X_pca"]
    else:
        source = adata.X

    if local_start is not None and local_end is not None:
        X_chunk = source[local_start:local_end]
    elif local_start is not None:
        X_chunk = source[local_start]
    else:
        X_chunk = source

    if hasattr(X_chunk, "toarray"):
        X_chunk = X_chunk.toarray()

    feat = np.array(X_chunk, dtype=np.float32)
    if flatten:
        feat = feat.flatten()
    return feat


def read_rank_gene_tokens(adata, local_start=None, local_end=None):
    if "rank" not in adata.obsm_keys():
        raise KeyError("H5AD is missing obsm['rank'] required for gene-token Q-Former input")
    source = adata.obsm["rank"]
    if local_start is not None and local_end is not None:
        chunk = source[local_start:local_end]
    elif local_start is not None:
        chunk = source[local_start]
    else:
        chunk = source
    return np.array(chunk)


# [统一单点事实] 所有的 Special Token IDs
SPECIAL_TOKEN_IDS = {
    "soc_id": 151669,
    "eoc_id": 151670,
}


def load_config(config_path: str) -> Dict:
    with open(config_path, "r") as f:
        return copy.deepcopy(json.load(f))

def resolve_ckpt_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_file():
        return path
    candidates = [
        path / "state.pt",
        path / "pytorch_model.bin",
        path / "model.pt",
        path / "checkpoint.pt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Cannot resolve checkpoint from: {path_str}")

def normalize_state_dict(raw_state: Dict) -> Dict:
    state = raw_state.get("model", raw_state) if isinstance(raw_state, dict) else raw_state
    clean_state = {}
    for k, v in state.items():
        new_k = k.replace("module.", "") if k.startswith("module.") else k
        clean_state[new_k] = v
    return clean_state

def resolve_dtype(dtype_name: str) -> torch.dtype:
    mapping = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    return mapping[dtype_name]

def _safe_torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def resolve_model_path(model_path: str) -> str:
    candidate = Path(model_path)
    if candidate.exists():
        return str(candidate)

    fallbacks = [
        Path('/data/bgi/data/projects/multimodal/zxy/zxy/models/qwen2.5-7b-instruct'),
        Path('/data/bgi/data/projects/multimodal/zyh/hf_cache/Qwen2.5-7B-Instruct'),
    ]
    for fb in fallbacks:
        if fb.exists():
            print(f"[resolve_model_path] configured llm_model_path not found: {model_path}")
            print(f"[resolve_model_path] fallback to local model path: {fb}")
            return str(fb)
    return model_path

def load_stage2_model(config_path: str, ckpt_path: str, device: str, dtype_name: str):
    import sys
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from src_ablation_cw.models.modeling_cell_transformer_for_sft_cw import CellTransformerForSFTCW

    config = load_config(config_path)
    model_path = resolve_model_path(config["model"]["llm_model_path"])
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = 151643

    for token_name in ("<|im_start|>", "<|im_end|>"):
        token_ids = tokenizer.encode(token_name, add_special_tokens=False)
        if len(token_ids) != 1:
            raise ValueError(f"Tokenizer does not treat {token_name} as a single special token: {token_ids}")

    model = CellTransformerForSFTCW(config, special_tokens_ids=SPECIAL_TOKEN_IDS)
    state = _safe_torch_load(resolve_ckpt_path(ckpt_path))
    clean_state_dict = normalize_state_dict(state)

    # Detect LoRA checkpoint and wrap showo accordingly
    is_lora_ckpt = any("lora_A" in k or "lora_B" in k for k in clean_state_dict.keys())
    if is_lora_ckpt:
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError:
            raise ImportError("LoRA checkpoint detected but peft is not installed.")

        cw_cfg = config.get("training", {}).get("cw_ablation", {})
        lora_cfg = LoraConfig(
            r=int(cw_cfg.get("lora_r", 8)),
            lora_alpha=int(cw_cfg.get("lora_alpha", 16)),
            lora_dropout=float(cw_cfg.get("lora_dropout", 0.05)),
            bias="none",
            target_modules=cw_cfg.get(
                "lora_target_modules",
                ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            ),
            task_type="CAUSAL_LM",
        )
        model.showo = get_peft_model(model.showo, lora_cfg)
        print(f"[load_stage2_model] LoRA checkpoint detected (r={lora_cfg.r}, alpha={lora_cfg.lora_alpha}). Wrapped showo with LoRA.")

    missing, unexpected = model.load_state_dict(clean_state_dict, strict=False)
    critical_missing = [k for k in missing if "showo.model.layers" in k or "cell_embedder" in k]
    if critical_missing:
         raise RuntimeError(f"加载权重失败！以下关键层缺失:\n{critical_missing[:5]}...")

    qformer_ckpt_keys = [k for k in clean_state_dict.keys() if "pathway_qformer" in k]
    pathway_embedding_ckpt_keys = [k for k in clean_state_dict.keys() if "pathway_embeddings" in k]
    qformer_missing = [k for k in missing if "pathway_qformer" in k]
    pathway_embedding_missing = [k for k in missing if "pathway_embeddings" in k]

    print(f"[load_stage2_model] missing_keys={len(missing)}, unexpected_keys={len(unexpected)}")
    print(
        f"[load_stage2_model] qformer_ckpt_keys={len(qformer_ckpt_keys)}, "
        f"pathway_embedding_ckpt_keys={len(pathway_embedding_ckpt_keys)}, "
        f"qformer_missing_after_load={len(qformer_missing)}, "
        f"pathway_embedding_missing_after_load={len(pathway_embedding_missing)}"
    )
    
    model.eval()
    target_dtype = resolve_dtype(dtype_name)
    if target_dtype != torch.float32:
        model = model.to(dtype=target_dtype)
    model = model.to(device)

    return config, tokenizer, model


def load_cells_by_ids(feature_path: str, cell_ids: List[str], config: Optional[Dict] = None) -> Dict[str, Dict]:
    feature_path_obj = Path(feature_path)
    dataset_cfg = (config or {}).get("dataset", {})

    # Preferred fast path: precomputed per-cell top1200 intersected gene JSON.
    if feature_path_obj.is_file() and feature_path_obj.suffix == ".json":
        payload = json.loads(feature_path_obj.read_text())
        items = payload.get("items", []) if isinstance(payload, dict) else payload
        model_cfg = (config or {}).get("model", {})
        static_gene_ckpt_path = model_cfg.get("static_gene_embedding_ckpt_path")
        if not static_gene_ckpt_path:
            raise ValueError("model.static_gene_embedding_ckpt_path is required for JSON-backed evaluation")
        bundle = load_static_gene_bundle(static_gene_ckpt_path)
        static_gene_embeddings = bundle["static_gene_embeddings"].numpy()
        gene_input_tokens = int(dataset_cfg.get("gene_input_tokens", 1200))
        hidden_dim = int(static_gene_embeddings.shape[1])

        cell_map = {}
        for item in items:
            cell_id = str(item["cell_id"])
            static_idx = np.asarray(item.get("top_static_gene_indices", []), dtype=np.int64)
            out = np.zeros((gene_input_tokens, hidden_dim), dtype=np.float32)
            if static_idx.size > 0:
                use_idx = static_idx[:gene_input_tokens]
                out[: len(use_idx)] = static_gene_embeddings[use_idx]
            cell_map[cell_id] = {"cell_features": out, "item": item}

        missing = []
        result: Dict[str, Dict] = {}
        for cid in cell_ids:
            cell_id = str(cid)
            record = cell_map.get(cell_id)
            if record is None:
                missing.append(cell_id)
                continue
            result[cell_id] = {"cell_features": record["cell_features"]}

        if missing:
            preview = missing[:10]
            raise KeyError(f"{len(missing)} cell ids not found in feature JSON, first few: {preview}")
        return result

    data_cfg = (config or {}).get("data", {})
    model_cfg = (config or {}).get("model", {})
    h5ad_paths = data_cfg.get("gene_h5ad_paths", [])
    static_gene_ckpt_path = model_cfg.get("static_gene_embedding_ckpt_path")
    gene_input_tokens = int(dataset_cfg.get("gene_input_tokens", 1200))
    gene_rank_order = str(dataset_cfg.get("gene_rank_order", "desc"))

    if not h5ad_paths:
        raise ValueError("config.data.gene_h5ad_paths is required for evaluation")
    if not static_gene_ckpt_path:
        raise ValueError("model.static_gene_embedding_ckpt_path is required for evaluation")

    registry = ExpressionH5ADRegistry(
        h5ad_paths=h5ad_paths,
        static_gene_ckpt_path=static_gene_ckpt_path,
        gene_input_tokens=gene_input_tokens,
        gene_rank_order=gene_rank_order,
    )

    result: Dict[str, Dict] = {}
    missing: List[str] = []
    for cid in cell_ids:
        cell_id = str(cid)
        try:
            result[cell_id] = {"cell_features": registry.get_gene_tokens(cell_id).numpy()}
        except KeyError:
            missing.append(cell_id)

    if missing:
        preview = missing[:10]
        raise KeyError(f"{len(missing)} cell ids not found in gene_h5ad_paths, first few: {preview}")
    return result

def load_conversations(json_paths: Sequence[str]) -> List[Dict]:
    items: List[Dict] = []
    for json_path in json_paths:
        with open(json_path, "r") as f:
            items.extend(json.load(f))
    return items

def extract_last_assistant_answer(conversations: Sequence[Dict]) -> str:
    for turn in reversed(conversations):
        if turn.get("from") == "gpt":
            return turn.get("value", "")
    raise ValueError("Conversation has no assistant answer")

def replace_last_assistant_answer(conversations: Sequence[Dict], new_answer: str) -> List[Dict]:
    updated = []
    replaced = False
    for turn in reversed(conversations):
        if not replaced and turn.get("from") == "gpt":
            updated.append({"from": "gpt", "value": new_answer})
            replaced = True
        else:
            updated.append(dict(turn))
    if not replaced:
        raise ValueError("Conversation has no assistant answer to replace")
    updated.reverse()
    return updated

def build_stage2_eval_sample(
    condition_cell: Dict,
    conversations: Sequence[Dict],
    tokenizer,
    config: Dict,
    last_answer_override: Optional[str] = None,
    include_system_metadata: bool = True,
    append_image_tag: bool = True,
):
    dataset_cfg = config.get("dataset", {})
    max_seq_len = dataset_cfg.get("max_seq_len", 1024)
    cell_feature_tokens = dataset_cfg.get("cell_feature_tokens", 8)
    cell_feature_dim = dataset_cfg.get("cell_feature_dim", 768)
    gene_input_tokens = dataset_cfg.get("gene_input_tokens", 1200)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 151643
    bos_id = getattr(tokenizer, "bos_token_id", None)

    if last_answer_override is not None:
        conversations = replace_last_assistant_answer(conversations, last_answer_override)

    assistant_indices = [i for i, turn in enumerate(conversations) if turn.get("from") == "gpt"]
    if not assistant_indices:
        raise ValueError("Conversation has no assistant turn")
    target_assistant_idx = assistant_indices[-1]

    input_ids: List[int] = []
    labels: List[int] = []

    if bos_id is not None:
        input_ids.append(bos_id)
        labels.append(-100)

    # [对齐] System Prompt：始终保留基础 system prompt，确保与训练一致
    system_prompt = "You are a helpful assistant."
    sys_tokens = tokenizer.encode(f"<|im_start|>system\n{system_prompt}<|im_end|>\n", add_special_tokens=False)
    input_ids.extend(sys_tokens)
    labels.extend([-100] * len(sys_tokens))

    cell_tokens = [SPECIAL_TOKEN_IDS["soc_id"]] + [pad_id] * cell_feature_tokens + [SPECIAL_TOKEN_IDS["eoc_id"]]
    cell_start_pos = 0
    is_first_user = True

    for idx, turn in enumerate(conversations):
        role = turn["from"]
        val = str(turn.get("value", ""))
        
        # 🚨 [关键清洗] LLaMA 垃圾格式屏蔽
        val = val.replace("[INST]", "").replace("[/INST]", "")
        val = val.replace("<<SYS>>", "").replace("<</SYS>>", "")
        val = val.replace("<s>", "").replace("</s>", "").strip()
        
        if role == "human":
            if is_first_user:
                val = val.replace("<image>", "").strip()
                if append_image_tag and val:
                    val = "\n" + val

            prefix_tokens = tokenizer.encode("<|im_start|>user\n", add_special_tokens=False)
            input_ids.extend(prefix_tokens)
            labels.extend([-100] * len(prefix_tokens))

            if is_first_user:
                soc_pos = len(input_ids)
                input_ids.extend(cell_tokens)
                labels.extend([-100] * len(cell_tokens))
                cell_start_pos = soc_pos + 1
                is_first_user = False

            content_tokens = tokenizer.encode(f"{val}<|im_end|>\n", add_special_tokens=False)
            input_ids.extend(content_tokens)
            labels.extend([-100] * len(content_tokens))
            
        elif role == "gpt":
            prefix_tokens = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
            content_tokens = tokenizer.encode(f"{val}<|im_end|>\n", add_special_tokens=False)
            input_ids.extend(prefix_tokens + content_tokens)
            
            if idx == target_assistant_idx:
                labels.extend([-100] * len(prefix_tokens) + content_tokens)
            else:
                labels.extend([-100] * (len(prefix_tokens) + len(content_tokens)))

    if len(input_ids) > max_seq_len:
        input_ids = input_ids[:max_seq_len]
        labels = labels[:max_seq_len]

    input_ids_tensor = torch.tensor(input_ids, dtype=torch.long)
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    attention_mask = torch.ones_like(input_ids_tensor, dtype=torch.bool)
    seq_len = len(input_ids_tensor)

    cell_len = cell_feature_tokens if cell_start_pos > 0 and (cell_start_pos + cell_feature_tokens) <= seq_len else 0
    if cell_len == 0:
        cell_start_pos = 0

    raw_feature = torch.tensor(condition_cell["cell_features"], dtype=torch.float32)
    if raw_feature.dim() == 1:
        if raw_feature.shape[0] < cell_feature_dim:
            pad_size = cell_feature_dim - raw_feature.shape[0]
            raw_feature = torch.nn.functional.pad(raw_feature, (0, pad_size))
        elif raw_feature.shape[0] > cell_feature_dim:
            raw_feature = raw_feature[:cell_feature_dim]
    elif raw_feature.dim() == 2:
        token_count, feat_dim = raw_feature.shape
        if token_count < gene_input_tokens:
            raw_feature = torch.nn.functional.pad(raw_feature, (0, 0, 0, gene_input_tokens - token_count))
        elif token_count > gene_input_tokens:
            raw_feature = raw_feature[:gene_input_tokens]
        if feat_dim < cell_feature_dim:
            raw_feature = torch.nn.functional.pad(raw_feature, (0, cell_feature_dim - feat_dim))
        elif feat_dim > cell_feature_dim:
            raw_feature = raw_feature[:, :cell_feature_dim]
    else:
        raise ValueError(f"Unsupported eval cell_features ndim={raw_feature.dim()}")

    processed_cell_feature = raw_feature

    return {
        "input_ids": input_ids_tensor,
        "labels": labels_tensor,
        "attention_mask": attention_mask,
        "cell_features": processed_cell_feature,
        "cell_positions": torch.tensor([cell_start_pos, cell_len], dtype=torch.long),
        "modality_positions": torch.tensor([[0, 0]], dtype=torch.long),
        "gene_mask": torch.zeros(0, dtype=torch.bool),
    }

def collate_eval_batch(samples: Sequence[Dict], pad_token_id: int) -> Dict[str, torch.Tensor]:
    batch_size = len(samples)
    max_len = max(sample["input_ids"].shape[0] for sample in samples)

    input_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
    labels = torch.full((batch_size, max_len), -100, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.bool) # 🚀 必须为 bool
    cell_features = torch.stack([sample["cell_features"] for sample in samples], dim=0)
    cell_positions = torch.stack([sample["cell_positions"] for sample in samples], dim=0)
    modality_positions = torch.stack([sample["modality_positions"] for sample in samples], dim=0)
    gene_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)

    for i, sample in enumerate(samples):
        length = sample["input_ids"].shape[0]
        input_ids[i, :length] = sample["input_ids"]
        labels[i, :length] = sample["labels"]
        attention_mask[i, :length] = sample["attention_mask"]
        local_gene_mask = sample["gene_mask"]
        if local_gene_mask.numel() > 0:
            gene_mask[i, : local_gene_mask.shape[0]] = local_gene_mask

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "cell_features": cell_features,
        "cell_positions": cell_positions,
        "modality_positions": modality_positions,
        "gene_mask": gene_mask,
    }

@torch.no_grad()
def compute_response_ppl_explicit_mask(model, batch: Dict[str, torch.Tensor], device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    # 强制重定向至天然的自回归前向传播，彻底封杀旧版 4D mask 干扰
    return compute_response_ppl(model, batch, device)

@torch.no_grad()
def compute_response_ppl(model, batch: Dict[str, torch.Tensor], device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        cell_features=batch["cell_features"],
        cell_positions=batch["cell_positions"],
        modality_positions=batch.get("modality_positions"),
        labels=None,
    )
    
    logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs.logits

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = batch["labels"][..., 1:].contiguous()
    token_loss = F.cross_entropy(
        shift_logits.float().view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="none",
    ).view(shift_labels.shape)
    
    valid_mask = shift_labels != -100
    token_loss = torch.where(valid_mask, token_loss, torch.zeros_like(token_loss))
    example_loss = token_loss.sum(dim=1)
    example_len = valid_mask.sum(dim=1).clamp_min(1)
    example_ppl = torch.exp(example_loss / example_len)
    return example_ppl.cpu(), example_len.cpu()

def save_json(path: str, payload: Dict) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

def save_csv(path: str, rows: List[Dict], fieldnames: Sequence[str]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)