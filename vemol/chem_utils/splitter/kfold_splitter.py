
import numpy as np
import pandas as pd 
from sklearn.model_selection import KFold

from vemol.chem_utils.dataset_path import BENCHMARK_NAME, BENCHMARK_BASE_PATH

from vemol.chem_utils.splitter.scaffold_splitter import ScaffoldSplitter

class KfoldSplitter(ScaffoldSplitter):
    def __init__(self, dataset: str):
        super().__init__(dataset)
        self.num_samples = len(self.smiles)
    
    def split_datset(self, n_splits: int, start_split_id: int=0, random_seed: int=1):
        indices = np.arange(self.num_samples)
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)  
        split_indices = [(train_idx, valid_idx) for train_idx, valid_idx in kf.split(indices)]
        for i, (train_idx, valid_idx) in enumerate(split_indices):
            split_folder = self.dataset_folder / "splits"
            split_folder.mkdir(exist_ok=True, parents=True)
            split_file = split_folder / f"kfold-{start_split_id+i}.npy"
            if split_file.exists():
                print(f"Split file {split_file} already exists!")
                continue
            else:
                self.save_scaffold_split(split_file, train_idx, valid_idx, valid_idx)
        
        
DATASETS = ['polymer_egc', 'polymer_egb', 'polymer_eea', 'polymer_ei',
             'polymer_xc', 'polymer_eps', 'polymer_nc', 'polymer_eat']


if __name__ == "__main__":
    n_splits = 5
    for dataset in DATASETS:
        splitter = KfoldSplitter(dataset)
        splitter.split_datset(n_splits=n_splits, start_split_id=10, random_seed=20)
        print(f"Dataset {dataset} split done")