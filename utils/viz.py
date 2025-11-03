from typing import Optional, Dict, List

import matplotlib.pyplot as plt
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


def plot_embeddings(embeddings: np.ndarray, labels: Optional[np.ndarray] = None, method: str = "tsne"):
    if method == "pca":
        reducer = PCA(n_components=2)
    else:
        reducer = TSNE(n_components=2, init="pca", learning_rate="auto")
    reduced = reducer.fit_transform(embeddings)
    plt.figure(figsize=(6, 5))
    if labels is not None and labels.ndim == 1:
        scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap="viridis", s=8, alpha=0.8)
        plt.colorbar(scatter)
    else:
        plt.scatter(reduced[:, 0], reduced[:, 1], s=8, alpha=0.8)
    plt.title(f"Embedding visualization ({method.upper()})")
    plt.tight_layout()
    return plt.gcf()


def draw_molecule_from_smiles(smiles: str, size=(300, 200)):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Draw.MolToImage(mol, size=size)


def plot_training_curves(history: Dict[str, List[float]]):
    plt.figure(figsize=(7, 4))
    if "train_loss" in history:
        plt.plot(history["train_loss"], label="train_loss")
    if "val_loss" in history:
        plt.plot(history["val_loss"], label="val_loss")
    if "metric" in history:
        plt.plot(history["metric"], label="metric")
    plt.xlabel("Epoch")
    plt.legend()
    plt.tight_layout()
    return plt.gcf()


def plot_true_vs_pred(y_true: np.ndarray, y_pred: np.ndarray, title: str = "True vs Pred"):
    plt.figure(figsize=(5, 5))
    plt.scatter(y_true.flatten(), y_pred.flatten(), s=8, alpha=0.7)
    min_v = float(min(y_true.min(), y_pred.min()))
    max_v = float(max(y_true.max(), y_pred.max()))
    plt.plot([min_v, max_v], [min_v, max_v], "r--", linewidth=1)
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title(title)
    plt.tight_layout()
    return plt.gcf()


