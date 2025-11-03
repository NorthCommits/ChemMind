from dataclasses import dataclass
from typing import Dict, Any, Tuple, List, Optional

import os
import io
import gzip
import requests
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torch_geometric.data import Data as GeometricData
from torch_geometric.loader import DataLoader

from rdkit import Chem

from .featurization import smiles_to_mol, mol_to_graph


@dataclass
class DatasetMetadata:
    num_node_features: int
    num_edge_features: int
    num_tasks: int


class MoleculeDataset(Dataset):
    def __init__(self, smiles: List[str], targets: np.ndarray, task_type: str, num_edge_features: int) -> None:
        self.smiles = smiles
        self.targets = targets
        self.task_type = task_type
        self.num_edge_features = int(num_edge_features)

    def __len__(self) -> int:
        return len(self.smiles)

    def __getitem__(self, idx: int) -> GeometricData:
        s = self.smiles[idx]
        y = self.targets[idx]
        if isinstance(y, np.ndarray):
            if y.ndim == 0:
                y = y[None]
            if y.ndim == 1:
                y = y[None, :]
        mol = smiles_to_mol(s)
        if mol is None:
            mol = Chem.MolFromSmiles("C")
        x, edge_index, edge_attr, _ = mol_to_graph(mol)
        edge_attr_tensor = (
            torch.tensor(edge_attr, dtype=torch.float32)
            if edge_attr.size > 0
            else torch.zeros((0, self.num_edge_features), dtype=torch.float32)
        )
        data = GeometricData(
            x=torch.tensor(x, dtype=torch.float32),
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            edge_attr=edge_attr_tensor,
            y=torch.tensor(y, dtype=torch.float32),
        )
        return data


def load_dataset_from_csv(cfg: Dict[str, Any]) -> Tuple[List[str], np.ndarray]:
    path = cfg["csv"]["path"]
    smiles_col = cfg["csv"].get("smiles_column", "smiles")
    label_col = cfg["csv"].get("label_column", "label")
    df = pd.read_csv(path)
    smiles = df[smiles_col].astype(str).tolist()
    y = df[label_col].values
    if y.ndim == 1:
        y = y[:, None]
    return smiles, y.astype(np.float32)


def _ensure_file(path: str, url: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        with open(path, "wb") as f:
            f.write(resp.content)
    return path


def load_moleculenet(cfg: Dict[str, Any]) -> Tuple[List[str], np.ndarray]:
    dataset = cfg["dataset"].lower()
    data_dir = cfg["paths"].get("data_dir", "data")
    if dataset == "tox21":
        gz_path = os.path.join(data_dir, "tox21.csv.gz")
        url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz"
        _ensure_file(gz_path, url)
        with gzip.open(gz_path, "rt", encoding="utf-8") as f:
            df = pd.read_csv(f)
        smiles = df["smiles"].astype(str).tolist()
        task_cols = [c for c in df.columns if c not in ("smiles", "mol_id")]
        y = df[task_cols].values
        y = np.nan_to_num(y, nan=0.0)
        return smiles, y.astype(np.float32)
    if dataset in ("esol",):
        csv_path = os.path.join(data_dir, "delaney-processed.csv")
        url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv"
        _ensure_file(csv_path, url)
        df = pd.read_csv(csv_path)
        smiles = df["smiles"].astype(str).tolist()
        y = df[["measured log solubility in mols per litre"]].values.astype(np.float32)
        return smiles, y
    if dataset in ("lipo", "lipophilicity"):
        csv_path = os.path.join(data_dir, "Lipophilicity.csv")
        url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/Lipophilicity.csv"
        _ensure_file(csv_path, url)
        df = pd.read_csv(csv_path)
        smiles = df["smiles"].astype(str).tolist()
        y = df[["exp"]].values.astype(np.float32)
        return smiles, y
    raise ValueError(f"Unsupported MoleculeNet dataset: {dataset}")


def prepare_data(cfg: Dict[str, Any]) -> Tuple[MoleculeDataset, MoleculeDataset, MoleculeDataset, DatasetMetadata]:
    task_type = cfg["task"]
    if cfg["dataset"].lower() == "csv":
        smiles, y = load_dataset_from_csv(cfg)
    else:
        smiles, y = load_moleculenet(cfg)

    if task_type == "classification":
        y = (y > 0.5).astype(np.float32)

    # Filter out invalid SMILES early to prevent RDKit errors downstream
    valid_indices: List[int] = []
    valid_smiles: List[str] = []
    for i, s in enumerate(smiles):
        try:
            m = Chem.MolFromSmiles(s)
        except Exception:
            m = None
        if m is not None:
            valid_indices.append(i)
            valid_smiles.append(s)
    if len(valid_indices) < len(smiles):
        y = y[np.array(valid_indices)]
        smiles = valid_smiles

    stratify_vec: Optional[np.ndarray] = None
    if y.shape[1] == 1:
        if task_type == "classification":
            stratify_vec = y[:, 0]
    train_smiles, temp_smiles, train_y, temp_y = train_test_split(
        smiles, y, test_size=0.2, random_state=cfg.get("seed", 42), stratify=stratify_vec
    )
    val_smiles, test_smiles, val_y, test_y = train_test_split(
        temp_smiles, temp_y, test_size=0.5, random_state=cfg.get("seed", 42)
    )

    # Determine feature dimensions using a simple valid molecule
    dummy_mol = smiles_to_mol("CC")
    x_meta, _, edge_attr_meta, _ = mol_to_graph(dummy_mol)
    num_node_features = int(x_meta.shape[1])
    num_edge_features = int(edge_attr_meta.shape[1] if edge_attr_meta is not None and edge_attr_meta.size > 0 else 0)

    train_ds = MoleculeDataset(train_smiles, train_y, task_type, num_edge_features)
    val_ds = MoleculeDataset(val_smiles, val_y, task_type, num_edge_features)
    test_ds = MoleculeDataset(test_smiles, test_y, task_type, num_edge_features)

    metadata = DatasetMetadata(
        num_node_features=num_node_features,
        num_edge_features=num_edge_features,
        num_tasks=int(y.shape[1]),
    )
    return train_ds, val_ds, test_ds, metadata


def build_dataloaders(cfg: Dict[str, Any]):
    train_ds, val_ds, test_ds, meta = prepare_data(cfg)
    num_workers = int(cfg["training"].get("num_workers", 0))
    batch_size = int(cfg["training"]["batch_size"])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader, {
        "num_node_features": meta.num_node_features,
        "num_edge_features": meta.num_edge_features,
        "num_tasks": meta.num_tasks,
    }


