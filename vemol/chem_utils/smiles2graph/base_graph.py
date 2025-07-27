from typing import Optional

import dgl 
from rdkit import Chem
import torch

from vemol.chem_utils.featurizer.standard_featurizer import StandardFeaturizer
from vemol.chem_utils.graph_ops import extract_edges_from_mol
from vemol.chem_utils.utils import mol2smiles
from vemol.chem_utils.featurizer.standard_featurizer import ATOM_FEATURE_DIM, BOND_FEATURE_DIM


def smiles2dummy_graph(smiles: str):
    return dgl.graph(([], []), num_nodes=1)

def mol2base_graph(mol: Chem.Mol) -> dgl.DGLGraph:
    featurizer = StandardFeaturizer()
    
    atom_features = featurizer.featurize_atoms(mol)
    bond_features = featurizer.featurize_bonds(mol)
    
    edges = extract_edges_from_mol(mol)
    if len(edges) == 0:
        print(f"{mol2smiles(mol)} contains no bond!")
        src, tgt = [], [] 
    else:
        src, tgt = list(zip(*edges))
    
    graph = dgl.graph((src, tgt), num_nodes=mol.GetNumAtoms())
    graph.ndata['h'] = torch.FloatTensor(atom_features) 
    graph.edata['e'] = torch.FloatTensor(bond_features)
    
    return graph


def smiles2base_graph(smiles: str) -> Optional[dgl.DGLGraph]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"Invalid SMILES: {smiles}")
        # raise ValueError(f"Invalid SMILES: {smiles}")
        return None 
    graph = mol2base_graph(mol)
    return graph

if __name__ == "__main__":
    smiles = 'CC(=O)O[AlH3]OC(C)=O'
    mol = Chem.MolFromSmiles(smiles) 
    print(mol)
    