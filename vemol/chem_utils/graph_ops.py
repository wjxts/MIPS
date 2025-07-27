from typing import List, Tuple 

import rdkit 
from rdkit import Chem 

def extract_edges_from_mol(mol: Chem.rdchem.Mol, add_self_loop=False) -> List[Tuple[int, int]]:
    # 每条键返回两条无向边
    edges = []
    for edge in mol.GetBonds():
        i, j = edge.GetBeginAtomIdx(), edge.GetEndAtomIdx()
        edges.extend([(i, j), (j, i)])
    if add_self_loop:
        for i in range(mol.GetNumAtoms()):
            edges.append((i, i))
    return edges 

def extract_rings_from_mol(mol: Chem.rdchem.Mol) -> List[List[int]]:
    return [list(ring) for ring in mol.GetRingInfo().AtomRings()]

def extract_sssr_from_mol(mol: Chem.rdchem.Mol) -> List[List[int]]: 
    # 获取对称简化的最小环集
    return [list(ring) for ring in Chem.GetSymmSSSR(mol)]

