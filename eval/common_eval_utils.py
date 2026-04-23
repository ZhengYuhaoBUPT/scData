#!/usr/bin/env python3
# coding=utf-8

import copy
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import anndata as ad
import h5py
import torch
import torch.nn.functional as F

from src.models.sc_omni_attention import omni_attn_mask_vectorized
from transformers import AutoTokenizer

PROJECT_ROOT = Path("/mnt/c20250607/user/wanghaoran/zxy/zxy/zxy/project/sc_showo")

SPECIAL_TOKEN_IDS = {
    "soc_id": 151669,
    "eoc_id": 151670,
    "sog_id": 151665,
    "eog_id": 151666,
    "mask_gene_id": 151667,
}


def load_config(config_path: str) -> Dict:
    with open(config_path, "r") as f:
        config = json.load(f)
    config = copy.deepcopy(config)
    config.setdefault("sequence", {})
    config["sequence"].setdefault("gene_axis_ratio", 0.5)
    return config


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
    if isinstance(raw_state, dict) and "model" in raw_state and isinstance(raw_state["model"], dict):
        return raw_state["model"]
    return raw_state


def resolve_dtype(dtype_name: str) -> torch.dtype:
    mapping = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    return mapping[dtype_name]


def load_stage2_model(config_path: str, ckpt_path: str, device: str, dtype_name: str):
    import sys

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from src.models.modeling_gene_transformer_for_sft import GeneTransformer

    config = load_config(config_path)
    model_path = config["model"]["llm_model_path"]
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    for token_name in ("<|im_start|>", "<|im_end|>"):
        token_ids = tokenizer.encode(token_name, add_special_tokens=False)
        if len(token_ids) != 1:
            raise ValueError(
                f"Tokenizer does not treat {token_name} as a single special token: {token_ids}"
            )

    model = GeneTransformer(
        llm_vocab_size=tokenizer.vocab_size,
        llm_model_path=model_path,
        load_from_showo=False,
        config_dict=config,
        special_tokens_ids=SPECIAL_TOKEN_IDS,
    )

    state = torch.load(resolve_ckpt_path(ckpt_path), map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(normalize_state_dict(state), strict=False)
    print(f"[load_stage2_model] missing_keys={len(missing)}, unexpected_keys={len(unexpected)}")
    if missing:
        print(f"[load_stage2_model] first_missing={missing[:8]}")
    if unexpected:
        print(f"[load_stage2_model] first_unexpected={unexpected[:8]}")
    model.eval()

    target_dtype = resolve_dtype(dtype_name)
    if target_dtype != torch.float32:
        model = model.to(dtype=target_dtype)
    model = model.to(device)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = 151643
    return config, tokenizer, model


def iter_h5ad_paths(feature_path: str) -> List[Path]:
    path = Path(feature_path)
    if path.is_file():
        return [path]
    if path.is_dir():
        files = sorted(path.glob("*.h5ad"))
        if not files:
            raise FileNotFoundError(f"No .h5ad files found in: {path}")
        return files
    raise FileNotFoundError(f"Feature path not found: {feature_path}")


def safe_get(obs_dict: Dict, index: int, keys: List[str], default: str) -> str:
    for key in keys:
        if key in obs_dict and index < len(obs_dict[key]):
            value = obs_dict[key][index]
            if value is None:
                continue
            value_str = str(value)
            if value_str.strip() and value_str.lower() != "nan":
                return value_str
    return default


def build_system_prompt(cell: Dict) -> str:
    cell_id = str(cell["cell_id"])
    if "SRX" in cell_id:
        return "You are an AI assistant analyzing RNA-seq data and its corresponding gene sequence."

    obs_dict = cell["obs_dict"]
    local_idx = cell["obs_index"]
    tissue = safe_get(obs_dict, local_idx, ["tissue_definition", "tissue", "tissue_name"], "unknown tissue")
    disease = safe_get(obs_dict, local_idx, ["disease_definition", "disease", "disease_name"], "healthy condition")
    sex = safe_get(obs_dict, local_idx, ["sex_name", "sex"], "unknown sex")
    dev_stage = safe_get(obs_dict, local_idx, ["development_stage", "stage_name", "stage"], "unknown")
    return f"This is a single-cell sample from the {tissue} of a {disease} human {sex} at {dev_stage} developmental stage."


def load_cells_by_ids(feature_path: str, target_ids: Iterable[str]) -> Dict[str, Dict]:
    target_ids = {str(x) for x in target_ids}
    found: Dict[str, Dict] = {}

    for h5ad_path in iter_h5ad_paths(feature_path):
        if len(found) == len(target_ids):
            break

        adata_temp = ad.read_h5ad(h5ad_path, backed="r")
        obs_df = adata_temp.obs.copy()
        adata_temp.file.close()

        cell_ids = obs_df["cell_id"].tolist() if "cell_id" in obs_df else obs_df.index.tolist()
        matched_rows = [idx for idx, cell_id in enumerate(cell_ids) if str(cell_id) in target_ids and str(cell_id) not in found]
        if not matched_rows:
            continue

        obs_dict = obs_df.to_dict("list")
        with h5py.File(h5ad_path, "r") as h5_file:
            x_array = h5_file["X"][matched_rows].astype("float32")
            rank_array = h5_file["obsm"]["rank"][matched_rows].astype("int32")
            log1p_array = h5_file["obsm"]["rank_log1p"][matched_rows].astype("float32")

        for local_pos, global_row in enumerate(matched_rows):
            cell_id = str(cell_ids[global_row])
            found[cell_id] = {
                "cell_id": cell_id,
                "cell_features": x_array[local_pos],
                "rank_seq": rank_array[local_pos],
                "log1p": log1p_array[local_pos],
                "obs_dict": obs_dict,
                "obs_index": global_row,
                "h5ad_path": str(h5ad_path),
            }

    missing = sorted(target_ids - set(found.keys()))
    if missing:
        raise KeyError(f"Missing {len(missing)} cell ids from feature files. First few: {missing[:5]}")
    return found


def load_conversations(json_paths: Sequence[str]) -> List[Dict]:
    items: List[Dict] = []
    for json_path in json_paths:
        with open(json_path, "r") as f:
            data = json.load(f)
        items.extend(data)
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
):
    dataset_cfg = config["dataset"]
    max_seq_len = dataset_cfg.get("max_seq_len", 2048)
    cell_feature_tokens = dataset_cfg.get("cell_feature_tokens", 8)
    max_genes = dataset_cfg.get("max_genes", 1200)
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

    if include_system_metadata:
        system_prompt = build_system_prompt(condition_cell)
    else:
        system_prompt = "You are an AI assistant analyzing RNA-seq data and its corresponding gene sequence."
    sys_tokens = tokenizer.encode(f"<|im_start|>system\n{system_prompt}<|im_end|>\n", add_special_tokens=False)
    input_ids.extend(sys_tokens)
    labels.extend([-100] * len(sys_tokens))

    rank_seq = condition_cell["rank_seq"].tolist()
    gene_tokens = [SPECIAL_TOKEN_IDS["sog_id"]] + rank_seq + [SPECIAL_TOKEN_IDS["eog_id"]]
    cell_tokens = [SPECIAL_TOKEN_IDS["soc_id"]] + [pad_id] * cell_feature_tokens + [SPECIAL_TOKEN_IDS["eoc_id"]]

    gene_start_pos = 0
    cell_start_pos = 0
    sog_pos = 0
    eog_pos = 0
    soc_pos = 0
    eoc_pos = 0
    is_first_user = True

    for idx, turn in enumerate(conversations):
        role = turn["from"]
        value = turn.get("value", "")
        if role == "human":
            text = value.replace("<image>", "given this cell (with its gene sequence)")
            user_tokens = tokenizer.encode(f"<|im_start|>user\n{text}<|im_end|>\n", add_special_tokens=False)
            input_ids.extend(user_tokens)
            labels.extend([-100] * len(user_tokens))
            if is_first_user:
                sog_pos = len(input_ids)
                input_ids.extend(gene_tokens)
                labels.extend([-100] * len(gene_tokens))
                gene_start_pos = sog_pos + 1
                eog_pos = sog_pos + len(gene_tokens) - 1

                soc_pos = len(input_ids)
                input_ids.extend(cell_tokens)
                labels.extend([-100] * len(cell_tokens))
                cell_start_pos = soc_pos + 1
                eoc_pos = soc_pos + len(cell_tokens) - 1
                is_first_user = False
        elif role == "gpt":
            prefix_tokens = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
            content_tokens = tokenizer.encode(f"{value}<|im_end|>\n", add_special_tokens=False)
            input_ids.extend(prefix_tokens + content_tokens)
            if idx == target_assistant_idx:
                labels.extend([-100] * len(prefix_tokens) + content_tokens)
            else:
                labels.extend([-100] * (len(prefix_tokens) + len(content_tokens)))
        else:
            raise ValueError(f"Unsupported role: {role}")

    if len(input_ids) > max_seq_len:
        input_ids = input_ids[:max_seq_len]
        labels = labels[:max_seq_len]

    input_ids_tensor = torch.tensor(input_ids, dtype=torch.long)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    if not (labels_tensor != -100).any():
        raise ValueError(
            "Target assistant answer was fully truncated; increase max_seq_len or shorten the prompt context."
        )
    attention_mask = torch.ones_like(input_ids_tensor)

    seq_len = len(input_ids_tensor)
    position_ids_tensor = torch.arange(seq_len, dtype=torch.long)

    gene_mask = torch.zeros(max_genes, dtype=torch.bool)

    return {
        "input_ids": input_ids_tensor,
        "labels": labels_tensor,
        "attention_mask": attention_mask,
        "position_ids": position_ids_tensor,
        "cell_features": torch.tensor(condition_cell["cell_features"], dtype=torch.float32),
        "cell_positions": torch.tensor([cell_start_pos, cell_feature_tokens], dtype=torch.long),
        "modality_positions": torch.tensor([[gene_start_pos, max_genes]], dtype=torch.long),
        "gene_mask": gene_mask,
    }


def collate_eval_batch(samples: Sequence[Dict], pad_token_id: int) -> Dict[str, torch.Tensor]:
    batch_size = len(samples)
    max_len = max(sample["input_ids"].shape[0] for sample in samples)

    input_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
    labels = torch.full((batch_size, max_len), -100, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
    position_ids = torch.zeros((batch_size, max_len), dtype=torch.long)
    cell_features = torch.stack([sample["cell_features"] for sample in samples], dim=0)
    cell_positions = torch.stack([sample["cell_positions"] for sample in samples], dim=0)
    modality_positions = torch.stack([sample["modality_positions"] for sample in samples], dim=0)
    gene_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)

    for i, sample in enumerate(samples):
        length = sample["input_ids"].shape[0]
        input_ids[i, :length] = sample["input_ids"]
        labels[i, :length] = sample["labels"]
        attention_mask[i, :length] = sample["attention_mask"]
        position_ids[i, :length] = sample["position_ids"]
        local_gene_mask = sample["gene_mask"]
        gene_mask[i, : local_gene_mask.shape[0]] = local_gene_mask

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "cell_features": cell_features,
        "cell_positions": cell_positions,
        "modality_positions": modality_positions,
        "gene_mask": gene_mask,
    }




def forward_with_explicit_mask(model, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    input_ids = batch["input_ids"]
    attention_mask = batch.get("attention_mask")
    position_ids = batch.get("position_ids")
    cell_features = batch.get("cell_features")
    cell_positions = batch.get("cell_positions")
    modality_positions = batch.get("modality_positions")
    gene_mask = batch.get("gene_mask")

    b, seq_len = input_ids.shape
    device = input_ids.device

    input_embeds = model.showo.model.embed_tokens(input_ids)

    if attention_mask is not None:
        attention_mask = model._to_model_dtype(attention_mask)

    if cell_features is not None and cell_positions is not None:
        cell_features = model._to_model_dtype(cell_features)
        cell_embeds = model.cell_embedder(cell_features).view(
            b, model.cell_feature_tokens, model.hidden_size
        )
        new_input_embeds = []
        for i in range(b):
            start_pos = cell_positions[i, 0].item()
            length = cell_positions[i, 1].item()
            end_pos = start_pos + length
            row = torch.cat([
                input_embeds[i, :start_pos],
                cell_embeds[i],
                input_embeds[i, end_pos:],
            ], dim=0)
            new_input_embeds.append(row)
        input_embeds = torch.stack(new_input_embeds)

    if modality_positions is not None:
        new_input_embeds = []
        for i in range(b):
            g_start = modality_positions[i, 0, 0].item()
            g_len = modality_positions[i, 0, 1].item()
            if g_len > 0:
                gene_tokens = input_ids[i, g_start:g_start + g_len].clone()
                if gene_mask is not None:
                    gene_tokens[gene_mask[i][:g_len]] = model.gene_mask_idx
                else:
                    gene_tokens[gene_tokens == model.mask_gene_id] = model.gene_mask_idx
                gene_embeds = model.gene_embedder(gene_tokens)
                row = torch.cat([
                    input_embeds[i, :g_start],
                    gene_embeds,
                    input_embeds[i, g_start + g_len:],
                ], dim=0)
                new_input_embeds.append(row)
            else:
                new_input_embeds.append(input_embeds[i])
        input_embeds = torch.stack(new_input_embeds)

    combined_modalities = []
    for i in range(b):
        mods = []
        if cell_positions is not None:
            mods.append((cell_positions[i, 0].item(), cell_positions[i, 1].item()))
        if modality_positions is not None:
            g_start = modality_positions[i, 0, 0].item()
            g_len = modality_positions[i, 0, 1].item()
            if g_len > 0:
                mods.append((g_start, g_len))
        combined_modalities.append(mods)

    omni_mask_binary = omni_attn_mask_vectorized(
        B=b,
        LEN=seq_len,
        modalities=combined_modalities,
        device=device,
        inverted=False,
    ).bool()

    if attention_mask is not None:
        omni_mask_binary &= attention_mask.unsqueeze(1).unsqueeze(2).bool()

    dtype = input_embeds.dtype
    omni_mask_4d = torch.zeros((b, 1, seq_len, seq_len), dtype=dtype, device=device)
    omni_mask_4d.masked_fill_(~omni_mask_binary, torch.finfo(dtype).min)

    outputs = model.showo(
        inputs_embeds=input_embeds,
        attention_mask=omni_mask_4d,
        position_ids=position_ids,
        return_dict=True,
    )
    return outputs.logits


@torch.no_grad()
def compute_response_ppl_explicit_mask(model, batch: Dict[str, torch.Tensor], device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    logits = forward_with_explicit_mask(model, batch)
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

@torch.no_grad()
def compute_response_ppl(model, batch: Dict[str, torch.Tensor], device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        position_ids=batch["position_ids"],
        cell_features=batch["cell_features"],
        cell_positions=batch["cell_positions"],
        modality_positions=batch["modality_positions"],
        gene_mask=batch["gene_mask"],
    )
    if isinstance(outputs, (tuple, list)):
        logits = outputs[0]
    else:
        logits = outputs.logits

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
