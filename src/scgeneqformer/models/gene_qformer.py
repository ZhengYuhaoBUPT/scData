#!/usr/bin/env python3
# coding: utf-8

from typing import Tuple

import torch
import torch.nn as nn


class CrossAttentionBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True, dropout=dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(query, key_value, key_value, need_weights=False)
        query = self.norm1(query + attn_out)
        return self.norm2(query + self.ffn(query))


class GeneExpressionConditioner(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.expr_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, static_gene_embeddings: torch.Tensor, cell_expr: torch.Tensor) -> torch.Tensor:
        expr_feat = self.expr_mlp(cell_expr.unsqueeze(-1))
        return static_gene_embeddings.unsqueeze(0) + expr_feat


class GeneQFormerModel(nn.Module):
    def __init__(self, hidden_dim: int, num_queries: int, num_heads: int, num_layers: int, num_genes: int):
        super().__init__()
        self.conditioner = GeneExpressionConditioner(hidden_dim)
        self.query_tokens = nn.Parameter(torch.randn(num_queries, hidden_dim) * 0.02)
        self.blocks = nn.ModuleList([CrossAttentionBlock(hidden_dim, num_heads) for _ in range(num_layers)])
        self.reconstruction_head = nn.Sequential(
            nn.Linear(num_queries * hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, num_genes),
        )

    def forward(self, static_gene_embeddings: torch.Tensor, cell_expr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        dynamic_tokens = self.conditioner(static_gene_embeddings, cell_expr)
        queries = self.query_tokens.unsqueeze(0).expand(cell_expr.size(0), -1, -1)
        for block in self.blocks:
            queries = block(queries, dynamic_tokens)
        recon = self.reconstruction_head(queries.reshape(queries.size(0), -1))
        return queries, recon


class PathwayCellFeatureQFormer(nn.Module):
    def __init__(self, hidden_dim: int, num_queries: int, num_heads: int, num_layers: int, out_dim: int, use_reconstruction_head: bool = True):
        super().__init__()
        self.use_reconstruction_head = use_reconstruction_head
        self.query_residual = nn.Parameter(torch.randn(num_queries, hidden_dim) * 0.02)
        self.cell_to_context = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.cell_to_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.blocks = nn.ModuleList([CrossAttentionBlock(hidden_dim, num_heads) for _ in range(num_layers)])
        if self.use_reconstruction_head:
            self.reconstruction_head = nn.Sequential(
                nn.Linear(num_queries * hidden_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Linear(hidden_dim * 2, out_dim),
            )
        else:
            self.reconstruction_head = None

    def forward(self, pathway_embeddings: torch.Tensor, cell_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if cell_features.dim() == 2:
            cell_context = self.cell_to_context(cell_features).unsqueeze(1)
            cell_gate = self.cell_to_gate(cell_features).unsqueeze(1)
            pathway_memory = pathway_embeddings.unsqueeze(0) * (1.0 + cell_gate) + cell_context
            queries = pathway_embeddings.unsqueeze(0) + self.query_residual.unsqueeze(0) + cell_context
        elif cell_features.dim() == 3:
            pathway_memory = cell_features
            queries = pathway_embeddings.unsqueeze(0) + self.query_residual.unsqueeze(0)
            queries = queries.expand(cell_features.size(0), -1, -1)
        else:
            raise ValueError(f"Unsupported cell_features ndim={cell_features.dim()}, expected 2 or 3")

        for block in self.blocks:
            queries = block(queries, pathway_memory)
        recon = None
        if self.reconstruction_head is not None:
            recon = self.reconstruction_head(queries.reshape(queries.size(0), -1))
        return queries, recon


class RankedGeneCellFeatureQFormer(nn.Module):
    def __init__(self, hidden_dim: int, num_queries: int, num_heads: int, num_layers: int, out_dim: int, top_rank_genes: int):
        super().__init__()
        self.top_rank_genes = top_rank_genes
        self.rank_embedding = nn.Embedding(top_rank_genes, hidden_dim)
        self.expr_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.cell_to_context = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.cell_to_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.use_reconstruction_head = use_reconstruction_head
        self.query_residual = nn.Parameter(torch.randn(num_queries, hidden_dim) * 0.02)
        self.blocks = nn.ModuleList([CrossAttentionBlock(hidden_dim, num_heads) for _ in range(num_layers)])
        if self.use_reconstruction_head:
            self.reconstruction_head = nn.Sequential(
                nn.Linear(num_queries * hidden_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Linear(hidden_dim * 2, out_dim),
            )
        else:
            self.reconstruction_head = None

    def build_ranked_memory(
        self,
        static_gene_embeddings: torch.Tensor,
        cell_expr: torch.Tensor,
        cell_features: torch.Tensor,
    ) -> torch.Tensor:
        topk = min(self.top_rank_genes, cell_expr.size(1))
        sorted_idx = torch.argsort(cell_expr, dim=1, descending=True)[:, :topk]
        gene_embed = static_gene_embeddings.unsqueeze(0).expand(cell_expr.size(0), -1, -1)
        selected_gene_embed = torch.gather(
            gene_embed,
            1,
            sorted_idx.unsqueeze(-1).expand(-1, -1, static_gene_embeddings.size(1)),
        )
        selected_expr = torch.gather(cell_expr, 1, sorted_idx)
        expr_feat = self.expr_mlp(selected_expr.unsqueeze(-1))
        rank_ids = torch.arange(topk, device=cell_expr.device).unsqueeze(0).expand(cell_expr.size(0), -1)
        rank_feat = self.rank_embedding(rank_ids)
        cell_context = self.cell_to_context(cell_features).unsqueeze(1)
        cell_gate = self.cell_to_gate(cell_features).unsqueeze(1)
        return selected_gene_embed * (1.0 + cell_gate) + expr_feat + rank_feat + cell_context

    def forward(
        self,
        pathway_embeddings: torch.Tensor,
        static_gene_embeddings: torch.Tensor,
        cell_features: torch.Tensor,
        cell_expr: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        memory = self.build_ranked_memory(static_gene_embeddings, cell_expr, cell_features)
        cell_context = self.cell_to_context(cell_features).unsqueeze(1)
        queries = pathway_embeddings.unsqueeze(0) + self.query_residual.unsqueeze(0) + cell_context
        for block in self.blocks:
            queries = block(queries, memory)
        recon = self.reconstruction_head(queries.reshape(queries.size(0), -1))
        return queries, recon
