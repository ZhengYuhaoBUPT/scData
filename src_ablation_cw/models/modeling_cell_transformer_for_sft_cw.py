from pathlib import Path
from typing import Any, Dict, Optional
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, Qwen2ForCausalLM

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scgeneqformer.models.gene_qformer import PathwayCellFeatureQFormer
from src_ablation_cw.datasets.gene_token_utils import build_pathway_embeddings_from_static_gene_ckpt


class CellTransformerForSFTCW(nn.Module):
    def __init__(self, config: Dict[str, Any], special_tokens_ids: Dict[str, int]):
        super().__init__()
        self.config_dict = config
        model_cfg = config.get("model", {})
        data_cfg = config.get("dataset", {})
        train_cfg = config.get("training", {})

        self.hidden_size = int(model_cfg.get("hidden_size", 3584))
        self.cell_feature_dim = int(data_cfg.get("cell_feature_dim", 768))
        self.cell_feature_tokens = int(data_cfg.get("cell_feature_tokens", 50))
        self.gradient_checkpointing = bool(train_cfg.get("gradient_checkpointing", True))
        self.use_pathway_cell_qformer = bool(model_cfg.get("use_pathway_cell_qformer", True))
        self.qformer_num_heads = int(model_cfg.get("qformer_num_heads", 8))
        self.qformer_num_layers = int(model_cfg.get("qformer_num_layers", 2))
        self.train_pathway_cell_qformer = bool(model_cfg.get("train_pathway_cell_qformer", True))
        self.pathway_qformer_ckpt_path = model_cfg.get("pathway_qformer_ckpt_path")
        self.pathway_json_path = model_cfg.get("pathway_json_path", str(PROJECT_ROOT / "datasets/pathway/pathway_anchor_genes.json"))
        self.static_gene_embedding_ckpt_path = model_cfg.get("static_gene_embedding_ckpt_path")
        self.init_pathway_embeddings_from_static_genes = bool(model_cfg.get("init_pathway_embeddings_from_static_genes", False))

        llm_path = model_cfg["llm_model_path"]
        local_only = bool(model_cfg.get("local_files_only", True))

        self.llm_config = AutoConfig.from_pretrained(llm_path, trust_remote_code=True, local_files_only=local_only)

        self.showo = Qwen2ForCausalLM.from_pretrained(
            llm_path,
            config=self.llm_config,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            local_files_only=local_only,
        )

        if self.use_pathway_cell_qformer:
            self.pathway_qformer = PathwayCellFeatureQFormer(
                hidden_dim=self.cell_feature_dim,
                num_queries=self.cell_feature_tokens,
                num_heads=self.qformer_num_heads,
                num_layers=self.qformer_num_layers,
                out_dim=self.cell_feature_dim,
                use_reconstruction_head=False,
            )
            self.pathway_embeddings = nn.Parameter(torch.randn(self.cell_feature_tokens, self.cell_feature_dim) * 0.02)
            self._maybe_init_pathway_embeddings_from_static_genes()
            self._maybe_load_pretrained_pathway_qformer_assets()
            self.cell_embedder = nn.Sequential(
                nn.Linear(self.cell_feature_dim, self.hidden_size),
                nn.GELU(),
                nn.Linear(self.hidden_size, self.hidden_size),
            )
        else:
            target_dim = self.hidden_size * self.cell_feature_tokens
            self.cell_embedder = nn.Sequential(
                nn.Linear(self.cell_feature_dim, target_dim),
                nn.GELU(),
                nn.Linear(target_dim, target_dim),
            )

        self.direct_token_projector = nn.Sequential(
            nn.Linear(self.cell_feature_dim, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )

        if self.gradient_checkpointing:
            self.showo.gradient_checkpointing_enable()

        self.soc_id = special_tokens_ids["soc_id"]
        self.eoc_id = special_tokens_ids["eoc_id"]
        self.dummy_float = nn.Parameter(torch.empty(0, dtype=torch.bfloat16), requires_grad=False)

    def _maybe_init_pathway_embeddings_from_static_genes(self):
        if not self.init_pathway_embeddings_from_static_genes:
            print("[Model] init_pathway_embeddings_from_static_genes=false. Keep random pathway embedding initialization.")
            return

        ckpt_path = self.static_gene_embedding_ckpt_path
        if not ckpt_path:
            print("[Model] static_gene_embedding_ckpt_path is empty. Keep random pathway embedding initialization.")
            return

        ckpt_file = Path(ckpt_path)
        pathway_json_file = Path(self.pathway_json_path)
        missing = [str(p) for p in [ckpt_file, pathway_json_file] if not p.exists()]
        if missing:
            print(f"[Model] Skip static-gene pathway init because files are missing: {missing}")
            return

        try:
            pathway_names, pathway_embeddings, pathway_gene_counts = build_pathway_embeddings_from_static_gene_ckpt(
                pathway_json_path=str(pathway_json_file),
                static_gene_ckpt_path=str(ckpt_file),
                num_queries=self.cell_feature_tokens,
            )
        except Exception as exc:
            print(f"[Model] Failed static-gene pathway init: {exc}. Keep random pathway embedding initialization.")
            return

        with torch.no_grad():
            self.pathway_embeddings.copy_(pathway_embeddings.float())

        covered = sum(c > 0 for c in pathway_gene_counts)
        avg_genes = sum(pathway_gene_counts) / max(1, covered)
        print(
            f"[Model] Initialized pathway embeddings from static gene embeddings: "
            f"covered_pathways={covered}/{self.cell_feature_tokens}, avg_genes_per_pathway={avg_genes:.2f}"
        )

    def _maybe_load_pretrained_pathway_qformer_assets(self):
        ckpt_path = self.pathway_qformer_ckpt_path
        if not ckpt_path:
            print("[Model] pathway_qformer_ckpt_path is empty. Keep current pathway embeddings and randomly initialized Q-Former weights.")
            return

        ckpt_file = Path(ckpt_path)
        if not ckpt_file.exists():
            print(f"[Model] pathway_qformer_ckpt_path not found: {ckpt_file}. Keep current pathway embeddings and random Q-Former weights.")
            return

        payload = torch.load(str(ckpt_file), map_location="cpu")
        pathway_embeddings = payload.get("pathway_embeddings_768d") if isinstance(payload, dict) else None
        if pathway_embeddings is not None:
            if tuple(pathway_embeddings.shape) != (self.cell_feature_tokens, self.cell_feature_dim):
                print(
                    "[Model] Skip loading pathway_embeddings_768d due to shape mismatch: "
                    f"expected {(self.cell_feature_tokens, self.cell_feature_dim)}, got {tuple(pathway_embeddings.shape)} from {ckpt_file}"
                )
            else:
                with torch.no_grad():
                    self.pathway_embeddings.copy_(pathway_embeddings.float())

        qformer_state = None
        if isinstance(payload, dict):
            qformer_state = payload.get("pathway_qformer_state_dict")
            if qformer_state is None:
                qformer_state = payload.get("model_state_dict")

        if qformer_state:
            missing, unexpected = self.pathway_qformer.load_state_dict(qformer_state, strict=False)
            print(
                f"[Model] Loaded compatible Q-Former weights from {ckpt_file} "
                f"(missing={len(missing)}, unexpected={len(unexpected)})."
            )
        else:
            print(f"[Model] No compatible Q-Former state_dict found in {ckpt_file}. Keep random initialization.")

    def freeze_llm_backbone(self):
        """冻结 LLM 主干，保留 cell/Q-Former 条件分支可按配置训练。"""
        for param in self.showo.parameters():
            param.requires_grad = False
        for module in [self.cell_embedder, self.direct_token_projector]:
            for param in module.parameters():
                param.requires_grad = True
        if self.use_pathway_cell_qformer:
            for param in self.pathway_qformer.parameters():
                param.requires_grad = self.train_pathway_cell_qformer
            self.pathway_embeddings.requires_grad = self.train_pathway_cell_qformer
        qformer_state = "trainable" if (self.use_pathway_cell_qformer and self.train_pathway_cell_qformer) else "frozen"
        print(f"[Model] showo backbone is frozen. Cell bridge is trainable. Q-Former branch is {qformer_state}.")

    def unfreeze_llm_backbone(self):
        for param in self.showo.parameters():
            param.requires_grad = True
        print("[Model] LLM backbone is unfrozen.")

    def _to_model_dtype(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(self.dummy_float.dtype)

    def _build_cell_embeds(self, cell_features: torch.Tensor) -> torch.Tensor:
        if cell_features.dim() == 2:
            if self.use_pathway_cell_qformer:
                qformer_queries, _ = self.pathway_qformer(
                    self.pathway_embeddings.to(device=cell_features.device, dtype=cell_features.dtype),
                    cell_features,
                )
                return self.cell_embedder(qformer_queries)
            flat_embeds = self.cell_embedder(cell_features)
            return flat_embeds.view(-1, self.cell_feature_tokens, self.hidden_size)
        if cell_features.dim() == 3:
            if self.use_pathway_cell_qformer:
                qformer_queries, _ = self.pathway_qformer(
                    self.pathway_embeddings.to(device=cell_features.device, dtype=cell_features.dtype),
                    cell_features,
                )
                return self.cell_embedder(qformer_queries)
            return self.direct_token_projector(cell_features)
        raise ValueError(f"Unsupported cell_features ndim={cell_features.dim()}, expected 2 or 3")

    def forward(
        self,
        input_ids: torch.LongTensor,
        cell_features: torch.FloatTensor,
        cell_positions: torch.LongTensor,
        labels: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        bsz, seq_len = input_ids.shape

        input_embeds = self.showo.get_input_embeddings()(input_ids)

        cell_features = self._to_model_dtype(cell_features)
        cell_embeds = self._build_cell_embeds(cell_features)

        rows = []
        for i in range(bsz):
            start = int(cell_positions[i, 0].item())
            clen = int(cell_positions[i, 1].item())
            if clen <= 0:
                rows.append(input_embeds[i])
                continue

            end = start + clen
            row = torch.cat([
                input_embeds[i, :start],
                cell_embeds[i, :clen],
                input_embeds[i, end:],
            ], dim=0)
            rows.append(row)

        input_embeds = torch.stack(rows)

        use_cache = kwargs.get("use_cache", False)
        outputs = self.showo(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=False,
            use_cache=use_cache,
        )
        logits = outputs.logits

        loss_ntp = (logits * 0.0).sum()
        if labels is not None:
            ntp_labels = labels.clone()
            for i in range(bsz):
                c_start = int(cell_positions[i, 0].item())
                c_len = int(cell_positions[i, 1].item())
                if c_len > 0:
                    c_end = min(c_start + c_len + 1, ntp_labels.shape[1])
                    ntp_labels[i, max(c_start - 1, 0):c_end] = -100

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = ntp_labels[..., 1:].contiguous()
            loss_ntp = F.cross_entropy(
                shift_logits.float().view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
                reduction="mean",
            )

        loss_gene = torch.zeros_like(loss_ntp)
        detailed = {
            "ntp_sft": float(loss_ntp.detach().item()) if labels is not None else 0.0,
            "ntp_stage1": 0.0,
            "gene_sft": 0.0,
            "gene_stage1": 0.0,
        }

        return logits, loss_ntp, loss_gene, detailed
