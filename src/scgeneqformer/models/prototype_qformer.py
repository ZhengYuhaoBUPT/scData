#!/usr/bin/env python3
# coding: utf-8

from typing import Tuple

import torch
import torch.nn as nn


class StaticPrototypeEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, static_prototypes: torch.Tensor) -> torch.Tensor:
        return self.net(static_prototypes)


class ExpressionConditioner(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.bias = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, static_gene_embeddings: torch.Tensor, cell_expr: torch.Tensor) -> torch.Tensor:
        gate = self.gate(cell_expr.unsqueeze(-1))
        bias = self.bias(cell_expr.unsqueeze(-1))
        dynamic = static_gene_embeddings.unsqueeze(0) * (1.0 + gate) + bias
        return self.dropout(dynamic)


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


class PrototypeQFormerModel(nn.Module):
    def __init__(self, num_genes: int, hidden_dim: int, num_queries: int, num_heads: int, num_layers: int, dropout: float = 0.1):
        super().__init__()
        self.prototype_encoder = StaticPrototypeEncoder(num_genes, hidden_dim, dropout=dropout)
        self.conditioner = ExpressionConditioner(hidden_dim, dropout=dropout)
        self.query_tokens = nn.Parameter(torch.randn(num_queries, hidden_dim) * 0.02)
        self.blocks = nn.ModuleList([CrossAttentionBlock(hidden_dim, num_heads, dropout=dropout) for _ in range(num_layers)])
        self.reconstruction_head = nn.Sequential(
            nn.Linear(num_queries * hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, num_genes),
        )

    def encode_static(self, static_prototypes: torch.Tensor) -> torch.Tensor:
        return self.prototype_encoder(static_prototypes)

    def forward(self, static_prototypes: torch.Tensor, cell_expr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        static_gene_embeddings = self.encode_static(static_prototypes)
        dynamic_tokens = self.conditioner(static_gene_embeddings, cell_expr)
        queries = self.query_tokens.unsqueeze(0).expand(cell_expr.size(0), -1, -1)
        for block in self.blocks:
            queries = block(queries, dynamic_tokens)
        recon = self.reconstruction_head(queries.reshape(queries.size(0), -1))
        return static_gene_embeddings, queries, recon
