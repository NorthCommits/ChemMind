from typing import List, Optional, Tuple

import numpy as np
from rdkit import Chem


def atom_features(atom: Chem.Atom) -> List[int]:
    """Return a robust set of atom-level features for node attributes."""
    symbol = atom.GetSymbol()
    symbol_list = [
        "C", "N", "O", "S", "F", "Si", "P", "Cl", "Br", "I", "H", "B", "Li", "Na", "K", "Ca",
    ]
    symbol_one_hot = [int(symbol == s) for s in symbol_list] + [int(symbol not in symbol_list)]

    degree = atom.GetDegree()
    degree_one_hot = [int(degree == d) for d in range(0, 6)] + [int(degree >= 6)]

    formal_charge = atom.GetFormalCharge()
    formal_charge_bucket = [int(formal_charge == c) for c in (-2, -1, 0, 1, 2)]

    hybridization = atom.GetHybridization()
    hyb_list = [Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2, Chem.rdchem.HybridizationType.SP3]
    hybrid_one_hot = [int(hybridization == h) for h in hyb_list] + [int(hybridization not in hyb_list)]

    aromatic = int(atom.GetIsAromatic())
    num_h = atom.GetTotalNumHs()
    num_h_one_hot = [int(num_h == h) for h in range(0, 5)] + [int(num_h >= 5)]

    chiral_tag = atom.GetChiralTag()
    chiral_one_hot = [
        int(chiral_tag == Chem.rdchem.ChiralType.CHI_UNSPECIFIED),
        int(chiral_tag == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW),
        int(chiral_tag == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW),
        int(chiral_tag == Chem.rdchem.ChiralType.CHI_OTHER),
    ]

    return symbol_one_hot + degree_one_hot + formal_charge_bucket + hybrid_one_hot + [aromatic] + num_h_one_hot + chiral_one_hot


def bond_features(bond: Optional[Chem.Bond]) -> List[int]:
    """Return edge attributes for a bond; if None, return zero features for self-loops."""
    if bond is None:
        return [0] * 10
    bt = bond.GetBondType()
    bond_type_one_hot = [
        int(bt == Chem.rdchem.BondType.SINGLE),
        int(bt == Chem.rdchem.BondType.DOUBLE),
        int(bt == Chem.rdchem.BondType.TRIPLE),
        int(bt == Chem.rdchem.BondType.AROMATIC),
    ]
    conj = int(bond.GetIsConjugated())
    in_ring = int(bond.IsInRing())
    stereo = bond.GetStereo()
    stereo_one_hot = [
        int(stereo == Chem.rdchem.BondStereo.STEREONONE),
        int(stereo == Chem.rdchem.BondStereo.STEREOANY),
        int(stereo == Chem.rdchem.BondStereo.STEREOZ),
        int(stereo == Chem.rdchem.BondStereo.STEREOE),
    ]
    return bond_type_one_hot + [conj, in_ring] + stereo_one_hot


def smiles_to_mol(smiles: str) -> Optional[Chem.Mol]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    Chem.Kekulize(mol, clearAromaticFlags=True)
    return mol


def mol_to_graph(mol: Chem.Mol) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert an RDKit Mol into (node_features, edge_index, edge_features, node_mask)."""
    node_features = [atom_features(atom) for atom in mol.GetAtoms()]
    node_features_array = np.asarray(node_features, dtype=np.float32)

    rows = []
    cols = []
    edge_attr = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bf = bond_features(bond)
        rows += [i, j]
        cols += [j, i]
        edge_attr += [bf, bf]

    edge_index = np.vstack([np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64)]) if rows else np.zeros((2, 0), dtype=np.int64)
    edge_attr_array = np.asarray(edge_attr, dtype=np.float32) if edge_attr else np.zeros((0, 10), dtype=np.float32)

    node_mask = np.ones((node_features_array.shape[0],), dtype=np.float32)
    return node_features_array, edge_index, edge_attr_array, node_mask


