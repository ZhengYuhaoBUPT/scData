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
        self.pathway_qformer_ckpt_path = model_cfg.get(
            "pathway_qformer_ckpt_path",
            str(PROJECT_ROOT / "outputs/training_runs/scgene_qformer_compact_cellfeat_ranked_v2/scgene_qformer_cellfeat_checkpoint.pt"),
        )

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
            )
            self.pathway_embeddings = nn.Parameter(torch.randn(self.cell_feature_tokens, self.cell_feature_dim) * 0.02)
            self._load_pretrained_pathway_qformer_assets()
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

    def _load_pretrained_pathway_qformer_assets(self):
        ckpt_path = self.pathway_qformer_ckpt_path
        if not ckpt_path:
            raise ValueError("Pathway Q-Former checkpoint path is required when use_pathway_cell_qformer=true")

        ckpt_file = Path(ckpt_path)
        if not ckpt_file.exists():
            raise FileNotFoundError(f"Pathway Q-Former checkpoint not found: {ckpt_file}")

        payload = torch.load(str(ckpt_file), map_location="cpu")
        pathway_embeddings = payload.get("pathway_embeddings_768d") if isinstance(payload, dict) else None
        if pathway_embeddings is None:
            raise KeyError(f"Checkpoint missing required key pathway_embeddings_768d: {ckpt_file}")

        if tuple(pathway_embeddings.shape) != (self.cell_feature_tokens, self.cell_feature_dim):
            raise ValueError(
                "Pathway embedding shape mismatch: "
                f"expected {(self.cell_feature_tokens, self.cell_feature_dim)}, got {tuple(pathway_embeddings.shape)} from {ckpt_file}"
            )

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
                f"[Model] Loaded pathway embeddings and compatible Q-Former weights from {ckpt_file} "
                f"(missing={len(missing)}, unexpected={len(unexpected)})."
            )
        else:
            raise KeyError(f"Checkpoint missing compatible Q-Former state_dict (expected pathway_qformer_state_dict or model_state_dict): {ckpt_file}")

    def freeze_llm_backbone(self):
        """冻结 LLM 与 Q-Former 分支，仅保留 token projector 可训练"""
        for param in self.showo.parameters():
            param.requires_grad = False
        for module in [self.cell_embedder, self.direct_token_projector]:
            for param in module.parameters():
                param.requires_grad = True
        if self.use_pathway_cell_qformer:
            for param in self.pathway_qformer.parameters():
                param.requires_grad = False
            self.pathway_embeddings.requires_grad = False
        print("[Model] showo backbone and Q-Former branch are frozen. Token projectors remain trainable.")

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
            if cell_features.size(1) != self.cell_feature_tokens:
                raise ValueError(
                    f"Expected cell_features second dim={self.cell_feature_tokens}, got {cell_features.size(1)}"
                )
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
