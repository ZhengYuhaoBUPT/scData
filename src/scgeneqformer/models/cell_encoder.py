#!/usr/bin/env python3
# coding: utf-8

import json
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

pad_token = "<pad>"
special_tokens = [pad_token, "<cls>", "<eoc>"]


class CellProjectionHead(nn.Module):
    def __init__(self, input_dim: int, intermediate_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, intermediate_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(intermediate_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.fc2(self.relu(self.fc1(x))))


def load_scgpt_modules(scgpt_path: str):
    modules_to_remove = [k for k in sys.modules.keys() if k.startswith("scgpt")]
    for mod in modules_to_remove:
        del sys.modules[mod]
    if scgpt_path not in sys.path:
        sys.path.insert(0, scgpt_path)
    from scgpt.model import TransformerModel  # type: ignore
    from scgpt.tokenizer.gene_tokenizer import GeneVocab  # type: ignore
    from scgpt.preprocess import binning  # type: ignore
    return TransformerModel, GeneVocab, binning


def load_cell_encoder(
    model_path: str,
    vocab_path: str,
    scgpt_path: str,
    device: torch.device,
) -> Tuple[nn.Module, nn.Module, Dict[str, int], callable]:
    TransformerModel, GeneVocab, binning = load_scgpt_modules(scgpt_path)
    vocab = GeneVocab.from_file(vocab_path)
    for s in special_tokens:
        if s not in vocab:
            vocab.append_token(s)

    model_dir = Path(model_path).parent
    with open(model_dir / "args.json", "r") as f:
        model_configs = json.load(f)

    gene_encoder = TransformerModel(
        ntoken=len(vocab),
        d_model=model_configs["embsize"],
        nhead=model_configs["nheads"],
        d_hid=model_configs["d_hid"],
        nlayers=model_configs["nlayers"],
        nlayers_cls=model_configs.get("n_layers_cls", 3),
        vocab=vocab,
        pad_token=pad_token,
        cell_emb_style=model_configs.get("cell_emb_style", "cls"),
        use_fast_transformer=True,
    )
    projection_head = CellProjectionHead(
        input_dim=model_configs["embsize"],
        intermediate_dim=model_configs.get("adapter_dim", 768),
        output_dim=768,
    )

    full_state_dict = torch.load(model_path, map_location="cpu")
    gene_encoder_state_dict = OrderedDict()
    for key, value in full_state_dict.items():
        if key.startswith("transformer_encoder_CL."):
            gene_encoder_state_dict[key.replace("transformer_encoder_CL.", "transformer_encoder.")] = value
        elif key.startswith(("transformer_encoder.", "cell2textAdapter.", "cls_decoder.", "decoder.", "mvc_decoder.")):
            continue
        else:
            gene_encoder_state_dict[key] = value
    gene_encoder.load_state_dict(gene_encoder_state_dict, strict=False)

    projection_head_state_dict = OrderedDict()
    for key, value in full_state_dict.items():
        if key.startswith("cell2textAdapter."):
            projection_head_state_dict[key.replace("cell2textAdapter.", "")] = value
    if projection_head_state_dict:
        projection_head.load_state_dict(projection_head_state_dict)

    gene_encoder.to(device).half().eval()
    projection_head.to(device).half().eval()
    vocab_dict = {token: vocab[token] for token in vocab.get_itos()}
    return gene_encoder, projection_head, vocab_dict, binning


def encode_pathway_vectors_to_cell_features(
    pathway_expr: torch.Tensor,
    genes: List[str],
    gene_to_scgpt_id: Dict[str, int],
    gene_encoder: nn.Module,
    projection_head: nn.Module,
    binning_fn,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    pad_id = gene_to_scgpt_id.get("<pad>", 0)
    gene_ids_np = np.asarray([gene_to_scgpt_id.get(g, pad_id) for g in genes], dtype=np.int64)
    features = []

    for start in range(0, pathway_expr.size(0), batch_size):
        batch_expr = pathway_expr[start:start + batch_size].cpu().numpy().astype(np.float32)
        binned = binning_fn(batch_expr, n_bins=51).astype(np.float32)
        expr_tensor = torch.from_numpy(binned).to(device=device, dtype=torch.float32)
        gene_tensor = torch.from_numpy(np.tile(gene_ids_np[None, :], (expr_tensor.size(0), 1))).to(device=device, dtype=torch.long)
        src_key_padding_mask = torch.zeros(expr_tensor.size(0), expr_tensor.size(1), dtype=torch.bool, device=device)
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                output = gene_encoder(
                    src=gene_tensor,
                    values=expr_tensor,
                    src_key_padding_mask=src_key_padding_mask,
                    CLS=False,
                    MVC=False,
                    ECS=False,
                )
                cell_emb = output["cell_emb"]
                cell_feat = projection_head(cell_emb)
        features.append(cell_feat.float().cpu())
    return torch.cat(features, dim=0)
