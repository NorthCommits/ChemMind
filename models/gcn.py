from typing import Literal

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool, global_add_pool, BatchNorm


class GCNModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.3,
        use_batch_norm: bool = True,
        pooling: Literal["mean", "sum", "max"] = "mean",
    ) -> None:
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList() if use_batch_norm else None
        self.convs.append(GCNConv(input_dim, hidden_dim))
        if use_batch_norm:
            self.bns.append(BatchNorm(hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            if use_batch_norm:
                self.bns.append(BatchNorm(hidden_dim))
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        self.pooling = pooling
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def pool(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        if self.pooling == "mean":
            return global_mean_pool(x, batch)
        if self.pooling == "sum":
            return global_add_pool(x, batch)
        return global_max_pool(x, batch)

    def forward(self, data) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.activation(x)
            if self.bns is not None:
                x = self.bns[i](x)
            x = self.dropout(x)
        x = self.pool(x, batch)
        out = self.head(x)
        return out


