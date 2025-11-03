import argparse
import io
import os
import sys
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import streamlit as st
import torch
import yaml

from rdkit import Chem
from rdkit.Chem import Descriptors, Draw

# Ensure project root is on sys.path for local imports when run via Streamlit
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.build import build_model
from utils.featurization import smiles_to_mol, mol_to_graph


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@st.cache_resource
def load_checkpoint(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    return ckpt


def predict_smiles(smiles: str, ckpt: Dict[str, Any], device: torch.device):
    metadata = ckpt["metadata"]
    config = ckpt["config"]
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
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    mol = smiles_to_mol(smiles)
    if mol is None:
        return None, None
    x, edge_index, edge_attr, _ = mol_to_graph(mol)
    import torch_geometric
    from torch_geometric.data import Data as GeometricData
    edge_attr_tensor = (
        torch.tensor(edge_attr, dtype=torch.float32)
        if edge_attr.size > 0
        else torch.zeros((0, int(metadata.get("num_edge_features", 0))), dtype=torch.float32)
    )
    data = GeometricData(
        x=torch.tensor(x, dtype=torch.float32),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        edge_attr=edge_attr_tensor,
        y=torch.zeros((1, metadata["num_tasks"]), dtype=torch.float32),
    )
    data.batch = torch.zeros(data.num_nodes, dtype=torch.long)
    data = data.to(device)
    with torch.inference_mode():
        logits = model(data)
    if ckpt["config"]["task"] == "classification":
        probs = torch.sigmoid(logits).cpu().numpy().flatten()
        return probs, "classification"
    preds = logits.cpu().numpy().flatten()
    return preds, "regression"


def main_streamlit():
    st.set_page_config(page_title="ChemMind", layout="centered")
    st.title("ChemMind – Molecular Property Prediction")

    with st.sidebar:
        st.header("Configuration")
        config_path = st.text_input("Config path", value="config.yaml")
        checkpoint_path = st.text_input("Checkpoint path", value="runs/latest/best.pt")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        st.caption(f"Device: {'CUDA' if device.type == 'cuda' else 'CPU'}")
        if os.path.exists(checkpoint_path):
            ckpt = load_checkpoint(checkpoint_path, device)
            st.success("Checkpoint loaded")
        else:
            ckpt = None
            st.warning("Checkpoint not found")

    tab_single, tab_batch = st.tabs(["Single", "Batch"])

    with tab_single:
        col1, col2 = st.columns([2, 1])
        with col1:
            smiles = st.text_input("SMILES", "CCO", help="Enter a valid SMILES string")
            run = st.button("Predict", type="primary")
        with col2:
            st.write("Examples")
            if st.button("Ethanol (CCO)"):
                smiles = "CCO"
            if st.button("Aspirin"):
                smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"

        if run:
            if ckpt is None:
                st.error("Load a valid checkpoint in the sidebar")
            else:
                preds, mode = predict_smiles(smiles, ckpt, device)
                if preds is None:
                    st.error("Invalid SMILES")
                else:
                    m = Chem.MolFromSmiles(smiles)
                    c1, c2 = st.columns([1, 1])
                    with c1:
                        if m is not None:
                            st.image(Draw.MolToImage(m), caption="Molecule")
                    with c2:
                        if m is not None:
                            mw = Descriptors.MolWt(m)
                            logp = Descriptors.MolLogP(m)
                            hbd = Descriptors.NumHDonors(m)
                            hba = Descriptors.NumHAcceptors(m)
                            st.markdown("**Descriptors**")
                            st.write({"MolWt": float(mw), "LogP": float(logp), "HBD": int(hbd), "HBA": int(hba)})
                    if mode == "classification":
                        st.markdown("**Predicted probabilities**")
                        chart_df = pd.DataFrame({"task": [f"task_{i}" for i in range(len(preds))], "prob": preds})
                        st.bar_chart(chart_df, x="task", y="prob", height=220)
                        st.json({f"task_{i}": float(p) for i, p in enumerate(preds)})
                    else:
                        st.markdown("**Predicted values**")
                        st.json({f"task_{i}": float(p) for i, p in enumerate(preds)})

    with tab_batch:
        st.write("Upload a CSV containing a 'smiles' column.")
        uploaded = st.file_uploader("CSV file", type=["csv"])
        if uploaded is not None:
            df = pd.read_csv(uploaded)
            smiles_col = st.selectbox("SMILES column", options=list(df.columns), index=list(df.columns).index("smiles") if "smiles" in df.columns else 0)
            if ckpt is None:
                st.error("Load a valid checkpoint in the sidebar")
            else:
                if st.button("Run batch predictions"):
                    records: List[Dict[str, Any]] = []
                    for s in df[smiles_col].astype(str).tolist():
                        preds, mode = predict_smiles(s, ckpt, device)
                        if preds is None:
                            rec = {"smiles": s, "error": "invalid"}
                        else:
                            rec = {"smiles": s}
                            for i, p in enumerate(preds):
                                rec[f"prediction_{i}"] = float(p)
                        records.append(rec)
                    out_df = pd.DataFrame(records)
                    st.dataframe(out_df, use_container_width=True)
                    csv_buf = out_df.to_csv(index=False).encode("utf-8")
                    st.download_button("Download results CSV", data=csv_buf, file_name="chemmind_predictions.csv", mime="text/csv")


if __name__ == "__main__":
    main_streamlit()


