import argparse
import os
import sys
from pathlib import Path

import pandas as pd

# Ensure project root on path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.data import load_moleculenet  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a large SMILES CSV from MoleculeNet datasets")
    parser.add_argument("--datasets", nargs="+", default=["tox21", "esol", "lipo"], help="Datasets to pull from")
    parser.add_argument("--limit", type=int, default=5000, help="Max number of rows in output")
    parser.add_argument("--out", type=str, default=str(PROJECT_ROOT / "data/chemmind_smiles_large.csv"), help="Output CSV path")
    args = parser.parse_args()

    all_smiles = []
    for ds in args.datasets:
        cfg = {
            "dataset": ds,
            "paths": {"data_dir": str(PROJECT_ROOT / "data")},
            "task": "classification",
            "seed": 42,
        }
        smiles, _ = load_moleculenet(cfg)
        all_smiles.extend(smiles)

    # Deduplicate while preserving order
    seen = set()
    unique_smiles = []
    for s in all_smiles:
        if s not in seen:
            seen.add(s)
            unique_smiles.append(s)
        if len(unique_smiles) >= args.limit:
            break

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"smiles": unique_smiles}).to_csv(out_path, index=False)
    print(f"Wrote {len(unique_smiles)} rows to {out_path}")


if __name__ == "__main__":
    main()


