
import numpy as np 

from dgllife.utils.featurizers import (ConcatFeaturizer, 
                                       bond_type_one_hot, bond_is_conjugated, bond_is_in_ring, bond_stereo_one_hot, 
                                       atomic_number_one_hot, atom_degree_one_hot, atom_formal_charge, 
                                       atom_num_radical_electrons_one_hot, atom_hybridization_one_hot, atom_is_aromatic, 
                                       atom_total_num_H_one_hot, atom_is_chiral_center, atom_chirality_type_one_hot, 
                                       atom_mass)
from functools import partial

import rdkit 
from rdkit import Chem 




ATOM_NUMS = 101
ATOM_FEATURE_DIM = 137
BOND_FEATURE_DIM = 14

SPATIAL_ATOM_NUMS = 10
ATOM_SPATIAL_FEATURE_DIM = 46

VIRTUAL_ATOM_FEATURE_PLACEHOLDER = -1
VIRTUAL_BOND_FEATURE_PLACEHOLDER = -1

INF = 100000000


bond_featurizer_all = ConcatFeaturizer([ # 14
    partial(bond_type_one_hot, encode_unknown=True), # 5 # 单/双/三/芳香/未知键
    bond_is_conjugated, # 1  # "C=C-C=C"中的所有键都是共轭键
    bond_is_in_ring, # 1 # 是否在环中
    partial(bond_stereo_one_hot, encode_unknown=True) # 7 # Chem.rdchem.BondStereo用于表示化学键的立体化学类型
    ])

atom_featurizer_all = ConcatFeaturizer([ # 137
    partial(atomic_number_one_hot, encode_unknown=True), #101
    partial(atom_degree_one_hot, encode_unknown=True), # 12
    atom_formal_charge, # 1
    partial(atom_num_radical_electrons_one_hot, encode_unknown=True), # 6  #"Radical electrons"指的是某个原子上的未配对电子, 一般来说是0
    partial(atom_hybridization_one_hot, encode_unknown=True), # 6
    atom_is_aromatic, # 1
    partial(atom_total_num_H_one_hot, encode_unknown=True), # 6
    atom_is_chiral_center, # 1
    atom_chirality_type_one_hot, # 2
    atom_mass, # 1
    ])

atom_featurizer_3d = ConcatFeaturizer([ # 137-(101-10) = 46
    partial(atomic_number_one_hot, allowable_set=list(range(1, 10)), encode_unknown=True), #10
    partial(atom_degree_one_hot, encode_unknown=True), # 12
    atom_formal_charge, # 1
    partial(atom_num_radical_electrons_one_hot, encode_unknown=True), # 6  #"Radical electrons"指的是某个原子上的未配对电子, 一般来说是0
    partial(atom_hybridization_one_hot, encode_unknown=True), # 6
    atom_is_aromatic, # 1
    partial(atom_total_num_H_one_hot, encode_unknown=True), # 6
    atom_is_chiral_center, # 1
    atom_chirality_type_one_hot, # 2
    atom_mass, # 1
    ])

class StandardFeaturizer:
    
    def __init__(self):
        pass
        
    def featurize_atoms(self, mol: Chem.rdchem.Mol) -> np.ndarray:
        atom_features = []
        for atom in mol.GetAtoms():
            atom_features.append(atom_featurizer_all(atom))
        return np.array(atom_features)
    
    def featurize_bonds(self, mol: Chem.rdchem.Mol) -> np.ndarray:
        bond_features = []
        for bond in mol.GetBonds():
            bond_feature = bond_featurizer_all(bond)
            bond_features.extend([bond_feature, bond_feature])
        return np.array(bond_features)


class SpatialFeaturizer(StandardFeaturizer):
    
    def __init__(self):
        super().__init__()
        
    def featurize_atoms(self, mol: Chem.rdchem.Mol) -> np.ndarray:
        atom_features = []
        for atom in mol.GetAtoms():
            atom_features.append(atom_featurizer_3d(atom))
        return np.array(atom_features)
    

if __name__ == "__main__":
    # smiles = 'CCC(NC(=O)c1scnc1C1CC1)C(=O)N1CCOCC1'
    # smiles = 'N1CCOCC1'
    smiles = 'C(C(F)(F)F)N'
    mol = Chem.MolFromSmiles(smiles)
    # featurizer = StandardFeaturizer()
    featurizer = SpatialFeaturizer()
    atom_features = featurizer.featurize_atoms(mol)
    bond_features = featurizer.featurize_bonds(mol)
    print(atom_features.shape, bond_features.shape)
    print(atom_features)
    
    
    
    