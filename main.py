import argparse
import os
import random
from typing import Dict, Any

import numpy as np
import torch
import yaml

from utils.training import train_and_evaluate
from utils.data import build_dataloaders
from utils.logging_utils import create_tensorboard_writer
from models.build import build_model


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ChemMind Training and Evaluation")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--task", type=str, choices=["classification", "regression"], help="Override task", default=None)
    parser.add_argument("--dataset", type=str, help="Override dataset name", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    if args.task is not None:
        config["task"] = args.task
    if args.dataset is not None:
        config["dataset"] = args.dataset

    os.makedirs(config["paths"]["runs_dir"], exist_ok=True)
    os.makedirs(config["paths"]["checkpoints_dir"], exist_ok=True)

    set_global_seed(config.get("seed", 42))

    device = torch.device(config.get("device", "cpu") if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader, metadata = build_dataloaders(config)

    model = build_model(
        model_name=config["model"]["name"],
        input_dim=metadata["num_node_features"],
        edge_dim=metadata.get("num_edge_features", 0),
        output_dim=metadata["num_tasks"],
        hidden_dim=config["model"]["hidden_dim"],
        num_layers=config["model"]["num_layers"],
        dropout=config["model"]["dropout"],
        use_batch_norm=config["model"].get("batch_norm", True),
        pooling=config["model"].get("pooling", "mean"),
    ).to(device)

    writer = create_tensorboard_writer(config["paths"]["runs_dir"]) 

    results = train_and_evaluate(
        model=model,
        loaders=(train_loader, val_loader, test_loader),
        config=config,
        device=device,
        writer=writer,
        task_type=config["task"],
        metadata=metadata,
    )

    print({k: float(v) if isinstance(v, (torch.Tensor, np.floating)) else v for k, v in results.items()})


if __name__ == "__main__":
    main()


