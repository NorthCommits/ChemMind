ChemMind – Graph Neural Network for Drug Discovery

Overview
ChemMind is an end-to-end AI system that converts molecules into graphs (atoms as nodes, bonds as edges) and uses Graph Neural Networks (GNNs) to predict molecular properties (e.g., toxicity, solubility, bioactivity). It provides a modular training pipeline, experiment logging, and a Streamlit app for inference and visualization.

Key Features
- Data preprocessing from SMILES using RDKit to PyTorch Geometric graphs
- GNN architectures: GCN and GIN with pooling, dropout, and batch normalization
- Unified training/evaluation for classification and regression
- Metrics: Accuracy, ROC-AUC, RMSE, MAE; TensorBoard logging; early stopping and LR scheduler
- Visualizations: training curves, embeddings (t-SNE/PCA), molecule rendering
- Web UI with Streamlit for inference from SMILES or CSV

Project Structure
```
ChemMind/
├── data/                 # Datasets (raw and processed)
├── models/               # Model architectures (GCN, GIN)
├── utils/                # Data prep, metrics, visualizations, training helpers
├── notebooks/            # Jupyter notebooks for experiments
├── app/                  # Streamlit web app
├── main.py               # Entry point for training and evaluation
├── requirements.txt      # Dependencies
├── config.yaml           # Hyperparameters and paths
└── README.md             # Documentation
```

Setup
1) Create and activate a Python 3.10+ environment.
2) Install dependencies (see requirements.txt). For PyTorch and PyG, prefer official instructions for your platform.
3) Optional: Use the Dockerfile to build a reproducible environment.

Datasets
- MoleculeNet: Tox21 (classification), ESOL/Lipophilicity (regression)
- Local CSVs: Provide `smiles` and `label`/`target` columns.

Quickstart
```bash
python main.py --config config.yaml --task classification --dataset tox21
python main.py --config config.yaml --task regression --dataset esol
```

Streamlit App
```bash
streamlit run app/app.py -- --config config.yaml --checkpoint runs/latest/best.pt
```

Reproducibility
- Configuration in `config.yaml`
- Seed control and deterministic flags
- TensorBoard logging in `runs/`

License
MIT


