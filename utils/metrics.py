from typing import Dict, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error, mean_absolute_error


def compute_classification_metrics(logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    probs = torch.sigmoid(logits).detach().cpu().numpy()
    preds = (probs >= 0.5).astype(np.float32)
    y_true = labels.detach().cpu().numpy()
    results = {}
    try:
        results["accuracy"] = float(accuracy_score(y_true.flatten(), preds.flatten()))
    except Exception:
        results["accuracy"] = float("nan")
    try:
        if y_true.shape[1] == 1:
            results["roc_auc"] = float(roc_auc_score(y_true, probs))
        else:
            aucs = []
            for i in range(y_true.shape[1]):
                yi = y_true[:, i]
                pi = probs[:, i]
                if len(np.unique(yi)) > 1:
                    aucs.append(roc_auc_score(yi, pi))
            results["roc_auc"] = float(np.mean(aucs)) if len(aucs) > 0 else float("nan")
    except Exception:
        results["roc_auc"] = float("nan")
    return results


def compute_regression_metrics(preds: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    y_pred = preds.detach().cpu().numpy()
    y_true = targets.detach().cpu().numpy()
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    return {"rmse": rmse, "mae": mae}


