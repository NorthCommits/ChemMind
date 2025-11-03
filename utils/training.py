from typing import Tuple, Dict, Any

import os
import math
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from torch_geometric.loader import DataLoader

from .metrics import compute_classification_metrics, compute_regression_metrics


def get_loss_fn(task_type: str) -> nn.Module:
    if task_type == "classification":
        return nn.BCEWithLogitsLoss()
    if task_type == "regression":
        return nn.MSELoss()
    raise ValueError(f"Unsupported task type: {task_type}")


def get_scheduler(optimizer: torch.optim.Optimizer, cfg: Dict[str, Any]):
    sched = cfg["training"].get("lr_scheduler", "cosine")
    if sched == "step":
        return StepLR(optimizer, step_size=cfg["training"].get("step_size", 20), gamma=cfg["training"].get("gamma", 0.5))
    if sched == "cosine":
        return CosineAnnealingLR(optimizer, T_max=cfg["training"].get("epochs", 100))
    return None


def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device, task_type: str):
    model.eval()
    all_logits = []
    all_targets = []
    with torch.inference_mode():
        for batch in loader:
            batch = batch.to(device)
            preds = model(batch)
            all_logits.append(preds.detach().cpu())
            all_targets.append(batch.y.detach().cpu())
    logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_targets, dim=0)
    if task_type == "classification":
        metrics = compute_classification_metrics(logits, targets)
        loss_fn = nn.BCEWithLogitsLoss()
        loss = loss_fn(logits, targets).item()
        metrics["val_loss"] = float(loss)
        return metrics
    metrics = compute_regression_metrics(logits, targets)
    loss_fn = nn.MSELoss()
    loss = loss_fn(logits, targets).item()
    metrics["val_loss"] = float(loss)
    return metrics


def train_and_evaluate(
    model: torch.nn.Module,
    loaders: Tuple[DataLoader, DataLoader, DataLoader],
    config: Dict[str, Any],
    device: torch.device,
    writer,
    task_type: str,
    metadata: Dict[str, Any],
):
    train_loader, val_loader, test_loader = loaders
    model.to(device)
    loss_fn = get_loss_fn(task_type)
    optimizer = Adam(model.parameters(), lr=config["training"]["lr"], weight_decay=config["training"].get("weight_decay", 0.0))
    scheduler = get_scheduler(optimizer, config)

    best_metric_value = math.inf if task_type == "regression" else -math.inf
    best_path = os.path.join(config["paths"]["checkpoints_dir"], "best.pt")
    patience = int(config["training"].get("early_stopping_patience", 15))
    patience_counter = 0

    global_step = 0
    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            preds = model(batch)
            loss = loss_fn(preds, batch.y)
            loss.backward()
            if config["training"].get("gradient_clip_norm", None):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config["training"]["gradient_clip_norm"])
            optimizer.step()
            writer.add_scalar("train/loss", loss.item(), global_step)
            running_loss += loss.item() * batch.num_graphs
            global_step += 1
        avg_train_loss = running_loss / len(train_loader.dataset)

        val_metrics = evaluate(model, val_loader, device, task_type)
        for k, v in val_metrics.items():
            writer.add_scalar(f"val/{k}", float(v), epoch)

        if scheduler is not None:
            scheduler.step()

        if task_type == "classification":
            current = val_metrics.get("roc_auc", -math.inf)
            improved = current > best_metric_value
        else:
            current = val_metrics.get("rmse", math.inf)
            improved = current < best_metric_value

        if improved:
            best_metric_value = current
            patience_counter = 0
            os.makedirs(os.path.dirname(best_path), exist_ok=True)
            torch.save({"model_state": model.state_dict(), "metadata": metadata, "config": config}, best_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    test_metrics = evaluate(model, test_loader, device, task_type)
    for k, v in test_metrics.items():
        writer.add_scalar(f"test/{k}", float(v), global_step)
    writer.flush()
    return {"best_val": best_metric_value, **{f"test_{k}": v for k, v in test_metrics.items()}, "checkpoint": best_path}


