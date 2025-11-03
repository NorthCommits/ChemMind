from typing import Literal

from .gcn import GCNModel
from .gin import GINModel


def build_model(
    model_name: Literal["gcn", "gin"],
    input_dim: int,
    edge_dim: int,
    output_dim: int,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
    use_batch_norm: bool,
    pooling: Literal["mean", "sum", "max"],
):
    if model_name == "gcn":
        return GCNModel(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            use_batch_norm=use_batch_norm,
            pooling=pooling,
        )
    if model_name == "gin":
        return GINModel(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            use_batch_norm=use_batch_norm,
            pooling=pooling,
        )
    raise ValueError(f"Unknown model name: {model_name}")


