from collections import defaultdict
from typing import List, Tuple 

import numpy as np
import pandas as pd 

from rdkit.Chem.Scaffolds import MurckoScaffold

from vemol.chem_utils.dataset_path import BENCHMARK_NAME, BENCHMARK_BASE_PATH


class ScaffoldSplitter:
    def __init__(self, dataset: str):
        benchmark_name = BENCHMARK_NAME[dataset]
        base_path = BENCHMARK_BASE_PATH[benchmark_name]
        self.dataset = dataset
        self.benchmark_name = benchmark_name
        self.base_path = base_path
        self.dataset_folder = base_path / dataset

        dataset_file = base_path / dataset / f"{dataset}.csv"
        df = pd.read_csv(dataset_file)
        
        self.smiles = df["smiles"].values.tolist()
    
    def generate_scaffold(self, smiles, include_chirality=False):
        """
        Obtain Bemis-Murcko scaffold from smiles
        :param smiles:
        :param include_chirality:
        :return: smiles of scaffold
        """
        # reduces each molecule to a scaffold by iteratively removing monovalent atoms until none remain
        # https://practicalcheminformatics.blogspot.com/2024/11/some-thoughts-on-splitting-chemical.html
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(
            smiles=smiles, includeChirality=include_chirality)
        return scaffold
    
    def save_scaffold_split(self, split_file: str, train_idx: List[int], valid_idx: List[int], test_idx: List[int]):
        split_dict = {'train': train_idx, 'valid':valid_idx, 'test': test_idx}
        new_split = np.array([np.array(split_dict['train']), 
                              np.array(split_dict['valid']), 
                              np.array(split_dict['test'])], dtype=object)
        print(f"Saving split to {split_file}")
        np.save(split_file, new_split)
        
        
    def split_datset(self, scaffold_id):
        split_file = self.dataset_folder / "splits" / f"scaffold-{scaffold_id}.npy"
        if split_file.exists():
            print(f"Split file {split_file} already exists!")
            return
        train_idx, valid_idx, test_idx = self.scaffold_split_ratio(self.smiles, random_seed=100+scaffold_id)
        if train_idx is None:
            print(f"fail to split {scaffold_id}!")
            return 
        self.save_scaffold_split(split_file, train_idx, valid_idx, test_idx)

    
    def split_datset_fix_train_size(self, scaffold_id: int, train_size: int):
        split_file = self.dataset_folder / "splits" / f"scaffold-{scaffold_id}.npy"
        if split_file.exists():
            print(f"Split file {split_file} already exists!")
            return
        # train_idx, valid_idx, test_idx = self.scaffold_split_fix_train_size(self.smiles, train_size, random_seed=200+scaffold_id)
        train_idx, valid_idx, test_idx = self.scaffold_split_fix_train_size(self.smiles, train_size, random_seed=280+scaffold_id)
        if train_idx is None:
            print(f"fail to split {scaffold_id}!")
            return 
        self.save_scaffold_split(split_file, train_idx, valid_idx, test_idx)
        

    def scaffold_split_fix_train_size(self, smiles_list: List[str], train_size: int, random_seed=12):
        assert train_size < len(smiles_list)
        valid_size = (len(smiles_list)-train_size) // 2
        return self.scaffold_split(smiles_list, train_size, valid_size, random_seed)
        
    def scaffold_split_ratio(self, smiles_list, frac_train=0.8, frac_valid=0.1, random_seed=12):
        train_size = int(frac_train * len(smiles_list))
        valid_size = int(frac_valid * len(smiles_list))
        return self.scaffold_split(smiles_list, train_size, valid_size, random_seed)
        
    def scaffold_split(self, smiles_list, 
                       train_size: int, 
                       valid_size: int, 
                       random_seed: int=12) -> Tuple[List[int], List[int], List[int]]:
        test_size = len(smiles_list) - train_size - valid_size
        
        all_scaffolds = defaultdict(list)
        for i, smiles in enumerate(smiles_list):
            scaffold = self.generate_scaffold(smiles, include_chirality=True)
            # print(smiles, scaffold)
            if scaffold not in all_scaffolds:
                all_scaffolds[scaffold] = [i]
            else:
                all_scaffolds[scaffold].append(i)
        all_scaffolds = {key: sorted(value) for key, value in all_scaffolds.items()}
        
        # print({k: len(all_scaffolds[k]) for k in all_scaffolds});exit()
        # Sort from largest to smallest scaffold sets
        all_scaffold_sets = [
            scaffold_set for (scaffold, scaffold_set) in sorted(
                all_scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)
        ]
        
        np.random.seed(random_seed)
        
        np.random.shuffle(all_scaffold_sets) # for random scaffold, optionally
        # print(all_scaffold_sets[:3])

        train_cutoff = train_size
        valid_cutoff = train_size + valid_size
        train_idx, valid_idx, test_idx = [], [], []
        for scaffold_set in all_scaffold_sets:
            if len(train_idx) + len(scaffold_set) > train_cutoff:
                if len(train_idx) + len(valid_idx) + len(scaffold_set) > valid_cutoff:
                    test_idx.extend(scaffold_set)
                else:
                    valid_idx.extend(scaffold_set)
            else:
                train_idx.extend(scaffold_set)

        # 划分不重叠
        assert len(set(train_idx).intersection(set(valid_idx))) == 0
        assert len(set(valid_idx).intersection(set(test_idx))) == 0
        assert len(set(train_idx).intersection(set(test_idx))) == 0
        
        # 划分大小和最初设计的size差异不能太大
        THRES = 5
        # THRES = 50
        try:
            assert abs(train_size - len(train_idx))<THRES
            assert abs(valid_size - len(valid_idx))<THRES
            assert abs(test_size - len(test_idx))<THRES
            
            # 特别用于esol, 因为scaffold dict太特殊了, 极其不均衡
            # assert len(train_idx)>50
            # assert len(valid_idx)>50
            # assert len(test_idx)>50
        except:
            print(f"train_size: {train_size}, len(train_idx): {len(train_idx)}")
            print(f"valid_size: {valid_size}, len(valid_idx): {len(valid_idx)}")
            print(f"test_size: {test_size}, len(test_idx): {len(test_idx)}")
            return None, None, None
        
        return train_idx, valid_idx, test_idx




            
if __name__ == "__main__":
    pass

    # dataset = 'polymer_egc'
    # scaffold_id = 3
    # scaffold_splitter = ScaffoldSplitter(dataset)
    # scaffold_splitter.split_datset(scaffold_id)
    # print("Done")