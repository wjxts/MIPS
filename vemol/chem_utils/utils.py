
from collections import defaultdict, Counter 
from pathlib import Path
from typing import Union, List

import hashlib
from joblib import Parallel, delayed
import numpy as np 
from rdkit import Chem
from tqdm import tqdm 


MAX_VALENCE = {'B': 3, 'Br':1, 'C':4, 'Cl':1, 'F':1, 'I':1, 'N':5, 'O':2, 'P':5, 'S':6} #, 'Se':4, 'Si':4}
CHEMBL_ATOMS = {'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Na': 11, 
                'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22, 
                'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30, 'Ga': 31, 'Ge': 32, 
                'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40, 'Mo': 42, 'Tc': 43, 
                'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50, 'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 
                'Ba': 56, 'La': 57, 'Nd': 60, 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Dy': 66, 'Yb': 70, 'Pt': 78, 'Au': 79, 
                'Hg': 80, 'Tl': 81, 'Pb': 82, 'Bi': 83, 'Ra': 88, 'Cf': 98}

def smiles2mol(smiles: str, kekulize=False, sanitize=True):
    '''turn smiles to molecule'''
    mol = Chem.MolFromSmiles(smiles, sanitize=sanitize)
    if kekulize:
        Chem.Kekulize(mol, True)
    return mol


def mol2smiles(mol, canonical=True):
    return Chem.MolToSmiles(mol, canonical=canonical)


def smiles_list2mol_list(smiles_list: List[str], n_jobs=1):
    if n_jobs==1:
        mol_list = [smiles2mol(smiles) for smiles in tqdm(smiles_list)]
    else:
        mol_list = Parallel(n_jobs=n_jobs)(delayed(smiles2mol)(smiles) for smiles in tqdm(smiles_list))
    return mol_list

def get_submol(mol, atom_indices, kekulize=False) -> Chem.Mol: 
    # 由index得到submol
    if len(atom_indices) == 1:
        atom_symbol = mol.GetAtomWithIdx(atom_indices[0]).GetSymbol()
        atom_symbol = f"[{atom_symbol}]"
        return smiles2mol(atom_symbol, kekulize)
    atom_id_set = set(atom_indices)
    edge_indices = []
    for i in range(mol.GetNumBonds()):
        bond = mol.GetBondWithIdx(i)
        begin_atom_id = bond.GetBeginAtomIdx()
        end_atom_id = bond.GetEndAtomIdx()
        if begin_atom_id in atom_id_set and end_atom_id in atom_id_set:
            edge_indices.append(i)
    mol = Chem.PathToSubmol(mol, edge_indices)
    return mol

# 目前似乎没用到
def get_submol_atom_map(mol, submol, group, kekulize=False):
    if len(group) == 1:
        return { group[0]: 0 }
    # turn to smiles order
    smi = mol2smiles(submol)
    submol = smiles2mol(smi, kekulize, sanitize=False)
    
    matches = mol.GetSubstructMatches(submol)
    old2new = { i: 0 for i in group }  # old atom idx to new atom idx
    found = False
    for m in matches:
        hit = True
        for i, atom_idx in enumerate(m):
            if atom_idx not in old2new:
                hit = False
                break
            old2new[atom_idx] = i
        if hit:
            found = True
            break
    assert found
    return old2new

def is_submol(mol, submol):
    return mol.HasSubstructMatch(submol)

def cnt_atom_from_mol(mol, return_dict=False):
    if return_dict:
        atom_list = [atom.GetSymbol() for atom in mol.GetAtoms()]
        return Counter(atom_list)
    else:
        return mol.GetNumAtoms()

def cnt_atom(smi, return_dict=False): #分子中每种原子有多少个，也可以直接用mol对象得到
    # 目前处理的不严格, 两字符原子只处理了Br和Cl
    
    # mol = smi2mol(smi)
    # if return_dict:
    #     atom_dict = defaultdict(int)    
    #     for atom in mol.GetAtoms():
    #         atom_dict[atom.GetSymbol()] += 1
    #     return atom_dict
    # else:
    #     return mol.GetNumAtoms()
    atom_dict = { atom: 0 for atom in CHEMBL_ATOMS}
    for i in range(len(smi)):
        symbol = smi[i].upper()
        next_char = smi[i+1] if i+1 < len(smi) else None
        if symbol == 'B' and next_char == 'r':
            symbol += next_char
        elif symbol == 'C' and next_char == 'l':
            symbol += next_char
        if symbol in atom_dict:
            atom_dict[symbol] += 1
    if return_dict:
        return atom_dict
    else:
        return sum(atom_dict.values())

from vemol.chem_utils.dataset_path import BENCHMARK_NAME, BENCHMARK_BASE_PATH


def get_smiles_from_pretrain_dataset(dataset: str = 'chembl29', N: int=1, random=True, seed: int=1):
    benchmark = BENCHMARK_NAME[dataset]
    benchmark_path = BENCHMARK_BASE_PATH[benchmark]
    file = benchmark_path / dataset / "smiles.smi"
    with open(file, 'r') as f:
        smiles_list = [s.strip() for s in f.readlines()]
    if not random:
        return smiles_list[:N]
    else:
        np.random.seed(seed)
        return np.random.choice(smiles_list, N, replace=False)
    
def encode_smiles(smiles: str):
    return hashlib.sha256(smiles.encode('ascii')).hexdigest().encode('ascii')

def load_smiles_from_file(file: Union[str, Path]):
    print(f'Loading mols from {file} ...')
    with open(file, 'r') as f:
        smiles_list = [s.strip() for s in f.readlines()]
    return smiles_list




EXAMPLE_SMILES = 'CCC(NC(=O)c1scnc1C1CC1)C(=O)N1CCOCC1'

if __name__ == "__main__":
    smiles_list = get_smiles_from_pretrain_dataset(N=3)
    print(smiles_list)