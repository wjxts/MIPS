from copy import deepcopy
from typing import List

import rdkit 
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import ChemicalFeatures
from rdkit.Chem.Pharm2D import Generate
from rdkit.Chem.Pharm2D.SigFactory import SigFactory
from rdkit.Chem.AtomPairs import Torsions
import numpy as np
from tqdm import tqdm 
# from multiprocessing import Pool
from functools import partial 
from joblib import Parallel, delayed

from vemol.chem_utils.fingerprint.descriptors.rdNormalizedDescriptors import RDKit2DNormalized


generator = RDKit2DNormalized()
# other fp
# topological_torsion
FP_FUNC_DICT = {}
FP_DIM = {"ecfp": 1024, # 1024
          "rdkfp": 1024,
          "maccs": 167, 
          "atom_pair": 1024, 
          "torsion": 1024, 
          'md': 200,
          'e3fp': 512,
          'e3fp_rb': 512,
          "pharm": 3348, 
          "torsion_3d": 512, 
          "atom_pair_3d": 512,
          "pharm_3d": 108,}

# FP_DIM = {"ecfp": 512,
#           "rdkfp": 512,
#           "maccs": 167, 
#           "atom_pair": 512, 
#           "torsion": 512, 
#           'md': 200,
#           'e3fp': 512,
#           'e3fp_rb': 512,
#           "pharm": 3348, }

def register_fp(name):
    def decorator(fp_func):
        FP_FUNC_DICT[name] = fp_func
        return fp_func
    return decorator 

@register_fp("ecfp")
def ecfp_fp(mol):
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=FP_DIM["ecfp"])
    fp = np.array(fp)
    assert len(fp) == FP_DIM["ecfp"]
    return fp
    
@register_fp("rdkfp")
def rdk_fp(mol):
    fp = Chem.RDKFingerprint(mol, minPath=1, maxPath=7, fpSize=FP_DIM["rdkfp"])
    fp = np.array(fp)
    assert len(fp) == FP_DIM["rdkfp"]
    return fp
    
@register_fp("maccs")
def maccs_fp(mol):
    fp = MACCSkeys.GenMACCSKeys(mol) 
    fp = np.array(fp) # shape: (167, )
    assert len(fp) == FP_DIM["maccs"]
    return fp 
    
@register_fp("atom_pair")
def atom_pair_fp(mol):
    fp = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=FP_DIM["atom_pair"])
    fp = np.array(fp)
    assert len(fp) == FP_DIM["atom_pair"]
    return fp 

@register_fp("atom_pair_3d")
def atom_pair_3d_fp(mol):
    fp = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=FP_DIM["atom_pair_3d"], use2D=False)
    fp = np.array(fp)
    assert len(fp) == FP_DIM["atom_pair_3d"]
    return fp 

def fp_to_np(fp, nbits):
    np_fps = np.zeros((nbits,))
    for k in fp:
        # np_fps[k % 1024] += v
        np_fps[k % nbits] = 1
    return np_fps
    
@register_fp("torsion")
def torsion_fp(mol):
    fp = Torsions.GetTopologicalTorsionFingerprintAsIds(mol) # 是hash到的整数
    np_fps = fp_to_np(fp, FP_DIM['torsion'])
    assert len(np_fps) == FP_DIM['torsion']
    return np_fps 


def get_torsion_quads_and_angles(mol, confId=0):
    torsion_data = []
 
    for bond in mol.GetBonds():
        atom2 = bond.GetBeginAtom()
        atom3 = bond.GetEndAtom()
        
        
        neighbors2 = [atom.GetIdx() for atom in atom2.GetNeighbors() if atom.GetIdx() != atom3.GetIdx()]
        neighbors3 = [atom.GetIdx() for atom in atom3.GetNeighbors() if atom.GetIdx() != atom2.GetIdx()]
        

        for atom1 in neighbors2:
            for atom4 in neighbors3:
                torsion_quad = (atom1, atom2.GetIdx(), atom3.GetIdx(), atom4)
                torsion_angle = Chem.rdMolTransforms.GetDihedralDeg(mol.GetConformer(confId),
                                                                    *torsion_quad)  # 扭转角计算
                torsion_data.append((torsion_quad, torsion_angle))
    
    return torsion_data


def hash_torsion_quad_with_angle(torsion_quad, torsion_angle):

    torsion_angle = ((torsion_angle+45)%360)//90

    hashed_value = hash((torsion_quad, torsion_angle))
    return hashed_value

def idx2arr(idx: List[int], nbits: int):
    fp = np.zeros(nbits)

    fp[idx] = 1
    return fp 

@register_fp("torsion_3d")
def torsion_3d_fp(mol):
    nbits = FP_DIM["torsion_3d"]
    
    torsion_data = get_torsion_quads_and_angles(mol)


    hashed_torsions = []
    for torsion_quad, torsion_angle in torsion_data:
        hashed_value = hash_torsion_quad_with_angle(torsion_quad, torsion_angle)
        hashed_torsions.append(hashed_value%nbits)
    fp = idx2arr(hashed_torsions, nbits)
    assert len(fp) == nbits
    return fp

    
@register_fp("pharm")
def pharmacophore_fp(mol):
    fdefName = rdkit.RDConfig.RDDataDir + '/BaseFeatures.fdef' # 2988 / 3348
    # fdefName = rdkit.RDConfig.RDDocsDir + '/Book/data/MinimalFeatures.fdef' # 990
    factory = ChemicalFeatures.BuildFeatureFactory(fdefName)

    # Create a SigFactory object
    # sigFactory = SigFactory(factory, minPointCount=2, maxPointCount=3) # for BaseFeatures.fdef
    sigFactory = SigFactory(factory, minPointCount=2, maxPointCount=3, trianglePruneBins=False) # For MinimalFeatures.fdef

    # Restrict the features that should be considered when generating the 2D pharmacophore fingerprints
    sigFactory.SetBins([(0, 2), (2, 5), (5, 8)])
    sigFactory.Init()
    # print(sigFactory.GetSigSize())
    # Generate the pharmacophore fingerprint
    fp = Generate.Gen2DFingerprint(mol, sigFactory)
    fp = np.array(fp)
    assert len(fp) == FP_DIM["pharm"]
    return fp 


@register_fp("pharm_3d")
def pharmacophore_3d_fp(mol):
    fdefName = rdkit.RDConfig.RDDataDir + '/BaseFeatures.fdef' 
    factory = ChemicalFeatures.BuildFeatureFactory(fdefName)
    sigFactory = SigFactory(factory, minPointCount=2, maxPointCount=2, trianglePruneBins=False) # For MinimalFeatures.fdef
    sigFactory.SetBins([(0, 2), (2, 5), (5, 8)])
    sigFactory.Init()
    fp = Generate.Gen2DFingerprint(mol, sigFactory, dMat=Chem.Get3DDistanceMatrix(mol))
    fp = np.array(fp)
    assert len(fp) == FP_DIM["pharm_3d"]
    return fp 


@register_fp("md")
def mol_descriptor(smiles):
    md = generator.process(smiles)
    if md is None:
        print(f"md descriptor failed for smiles: {smiles}")
        return np.zeros(FP_DIM["md"])
    else:
        md = np.array(md[1:])
    assert len(md) == FP_DIM["md"]
    return md 



def merge_idx_common(fprints):
    idx = set(fprints[0].indices)
    for fp in fprints[1:]:
        idx = idx & set(fp.indices)
    return list(idx)

def merge_idx_union(fprints):
    idx = set(fprints[0].indices)
    for fp in fprints[1:]:
        idx = idx | set(fp.indices)
    return list(idx)
    
try:
    from e3fp.pipeline import fprints_from_smiles
    

    @register_fp("e3fp")
    def e3_fp(smiles, nbits=FP_DIM["e3fp"]):
        # print(smiles)
        confgen_params = {'max_energy_diff': 20.0, 'first': 3, 'forcefield': 'mmff94'}
        # fprint_params = {'bits': nbits, 'radius_multiplier': 1.5, 'rdkit_invariants': True, 'level': 2}
        fprint_params = {'bits': nbits, 'radius_multiplier': 1.5, 'rdkit_invariants': True}
        try:
            fprints = fprints_from_smiles(smiles, "ritalin", confgen_params=confgen_params, fprint_params=fprint_params)
        except:
            print(f"e3fp is None for smiles: {smiles}")
            return np.zeros(FP_DIM["e3fp"])
        if isinstance(fprints, list) and len(fprints) > 0:
            indices = merge_idx_union(fprints)
            fp = idx2arr(indices, nbits)
            return fp
        else:
            print(f"e3fp is None for smiles: {smiles}")
            return np.zeros(FP_DIM["e3fp"])
except:
    pass 





def get_fp(smiles, fp_name="ecfp"):
    # print(smiles)
    assert fp_name in FP_FUNC_DICT, f"fp_name: {fp_name} not in FP_FUNC_DICT"
    if fp_name in ['md', 'e3fp', 'e3fp_rb']:
        return FP_FUNC_DICT[fp_name](smiles)
    else:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"mol is None for smiles: {smiles}")
            return np.zeros(FP_DIM[fp_name])
        try:
            if '3d' in fp_name:
                mol = Chem.MolFromSmiles(smiles)
                mol = Chem.AddHs(mol)
                AllChem.EmbedMolecule(mol)
                success = AllChem.MMFFOptimizeMolecule(mol, mmffVariant="MMFF94")
                mol = Chem.RemoveHs(mol)
        except:
            print(f"3d fp failed for smiles: {smiles}")
            return np.zeros(FP_DIM[fp_name])
        return FP_FUNC_DICT[fp_name](mol)



def get_batch_fp(smiless, fp_name, n_jobs=1):

    if n_jobs == 1:
        print(f"sequentially extracting fp={fp_name} , n_jobs=1")
        fp_list = [get_fp(smiles, fp_name) for smiles in tqdm(smiless)]
    else:
        print(f"parallel extracting fp={fp_name} , n_jobs={n_jobs}")
        # fp_func = partial(get_fp, fp_name=fp_name)
        # fp_list = list(Pool(n_jobs).imap(fp_func, tqdm(smiless)))
        fp_list = Parallel(n_jobs=n_jobs)(delayed(get_fp)(smiles, fp_name) for smiles in tqdm(smiless))
    return np.stack(fp_list, axis=0)




def get_full_fp(smiless):
    res = {}
    selected_fp_name = ['ecfp', 'rdkfp', 'maccs', 'atom_pair', 'torsion', 'md']
    for fp_name in selected_fp_name:
        res[fp_name] = get_batch_fp(smiless, fp_name)
    return res 


def disturb_discrete_fp(fp: np.ndarray, disturb_rate: float):
    if disturb_rate == 0:
        return fp
    fp = deepcopy(fp)
    mask = np.random.binomial(n=1, p=disturb_rate, size=fp.shape)
    fp[mask==1] = 1 - fp[mask==1]
    return fp, mask 
    
def disturb_continous_fp(fp: np.ndarray, disturb_rate: float):
    if disturb_rate == 0:
        return fp
    fp = deepcopy(fp)
    mask = np.random.binomial(n=1, p=disturb_rate, size=fp.shape)
    noise = np.random.rand(*fp.shape)
    fp[mask==1] = noise[mask==1]
    return fp, mask

def disturb_fp(fp: np.ndarray, fp_name:str, disturb_rate: float):
    if fp_name == 'md':
        return disturb_continous_fp(fp, disturb_rate)
    else:
        return disturb_discrete_fp(fp, disturb_rate)

def test():
    smiless = ["CCO", "CCN", "CCO"]
    fp_name_list = ["ecfp", "rdkfp", "maccs", "atom_pair", "pharm", "torsion", 'md']
    for fp_name in fp_name_list:
        fp = get_batch_fp(smiless, fp_name)
        print(fp_name, fp.shape)
        # print(fp)
        print("="*20)

def test_disturb():
    smiless = ["CCO", "CCN", "CCO"]
    fp_name_list = ["md", "ecfp", "rdkfp", "maccs", "atom_pair", "pharm", "torsion"][:3]
    for fp_name in fp_name_list:
        fp = get_batch_fp(smiless, fp_name)
        print(fp_name, fp.shape)
        print(fp[:2, :5])
        fpp, mask = disturb_fp(fp, fp_name, 0.5)
        print(fpp[:2, :5])
        print("="*20)

def test_e3fp():
    from vemol.chem_utils.polymer_utils import augment_polymer, psmiles_star_sub
   
    smiles = 'CCC(C)c1cccc(C#N)c1'
    
    from time import time
    import cProfile
    import pstats
    start_time = time()
  
    fp = get_fp(smiles, "e3fp")
    print(np.where(fp==1))
   
    
    end_time = time()
    print(f"time cost: {end_time-start_time:.2f}s")
    
if __name__ == "__main__":
    # test_disturb()
    # test_e3fp()
    from time import time
    start_time = time()
    smiles = 'CCC(C)c1cccc(C#N)c1'
    mol = Chem.MolFromSmiles(smiles)
   
    fp = get_fp(smiles, "e3fp")
    print(fp.sum())
    end_time = time()
    print(f"time cost: {end_time-start_time:.2f}s")
    
    
    
    
    
        