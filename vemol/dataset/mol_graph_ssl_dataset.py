
from dataclasses import dataclass, field
import json 
from pathlib import Path
from typing import List, Any, Literal, Callable, Type

import numpy as np 
from omegaconf import II, MISSING, DictConfig

from vemol.dataset import register_dataset
from vemol.dataset.base_dataset import BaseDataLoader
from vemol.dataset.mol_graph_dataset import BaseMolGraphDatasetConfig, BaseMolGraphDataset
from vemol.chem_utils.featurizer.standard_featurizer import ATOM_NUMS
from vemol.chem_utils.dataset_path import DATASET_SIZE

from vemol.chem_utils.smiles2graph.complete_graph import smiles2complete_graph, base_graph2complete_graph
from vemol.chem_utils.smiles2graph.polymer_base_graph import base_graph2polymer_base_graph



local_defaults = [{"mol_graph_form_cfg": "atom_graph"}, 
                  {"mol_graph_collator_cfg": "mae"},]

@dataclass
class MolGraphSSLDatasetConfig(BaseMolGraphDatasetConfig):
    # 用default参数告诉hydra那个变量在config的层次里, 来初始化当前层次的config (把当前节点当作局部根节点)
    defaults: List[Any] = field(default_factory=lambda: local_defaults)
    dataset: str = field(
        default='mol_graph_ssl', metadata={'help': "The class of dataset."}
    )
    name: str = 'chembl29'
    num_workers: int = 8
    
class MolGraphMPPSSLDataset(BaseMolGraphDataset):
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        if cfg.mol_graph_form_cfg.name == 'polymer_atom_graph':
            self.load_valid_indices()
        super().__init__(cfg)
        self.init_random_idx_mapping()
    
    def init_random_idx_mapping(self):
        np.random.seed(self.cfg.seed)
        self.idx_mapping = np.random.permutation(self.max_dataset_size) 
        # self.idx_mapping = np.arange(self.max_dataset_size)
        
    def init_graph_data(self):
        self.load_smiles()
        if self.cfg.use_cache_data:
            self.init_on_disk_dataset()
    
    def load_valid_indices(self):
        file_path = self.dataset_folder / 'valid_indice_cache.json'
        self.valid_index_list = json.load(open(file_path, 'r'))
        
    @property
    def max_dataset_size(self):
        if self.cfg.mol_graph_form_cfg.name == 'polymer_atom_graph':
            return len(self.valid_index_list)
        return DATASET_SIZE[self.dataset_name]
    
    @property
    def dataset_size(self):
        downsize = self.cfg.mol_graph_collator_cfg.downsize
        assert downsize <= self.max_dataset_size, f"downsize {downsize} should be less than the max dataset size {self.max_dataset_size}."
        return self.max_dataset_size if downsize < 0 else downsize
    
    
    def __len__(self):
        return self.dataset_size
    
    def __getitem__(self, idx):
        idx = self.idx_mapping[idx]
        if self.cfg.mol_graph_form_cfg.name == 'polymer_atom_graph':
            idx = self.valid_index_list[idx]
        if self.cfg.use_cache_data and self.cfg.mol_graph_form_cfg.name != 'fragment_graph':
            # fragment graph的cache机制在fragmentor里面
            graph = self.get_data_item_on_disk(idx)
            if self.cfg.mol_graph_form_cfg.name == 'polymer_atom_graph':
                # smiles = self.total_smiles_list[idx]
                op = self.cfg.mol_graph_form_cfg.op
                main_chain_embed = self.cfg.mol_graph_form_cfg.main_chain_embed
                graph = base_graph2polymer_base_graph(graph, op, main_chain_embed)
        else:
            graph = self.smiles2graph(self.total_smiles_list[idx])
        if self.cfg.completion and self.cfg.use_cache_data:
            graph = base_graph2complete_graph(graph, max_length=self.cfg.max_length)
            
        kvecs = {name: self.kvec_data[name][idx] for name in self.cfg.kvec_names}
        
        if self.cfg.add_knodes:
            graph = self.add_knodes_to_graph(graph, kvecs)
            
        return graph, kvecs

    @property
    def in_memory_graph(self) -> bool:
        return False
    
    @property
    def in_memory_kvec(self):
        return False 
        
    @property
    def target_dim(self):
        if self.cfg.mol_graph_collator_cfg.name == 'mae':
            if self.cfg.mol_graph_form_cfg.name == 'atom_graph':
                return ATOM_NUMS
            elif self.cfg.mol_graph_form_cfg.name == 'polymer_atom_graph':
                return ATOM_NUMS
            elif self.cfg.mol_graph_form_cfg.name == 'fragment_graph':
                return len(self.fragmentor)
            else:
                raise ValueError(f"graph form {self.cfg.mol_graph_form_cfg.name} not supported.")
        elif self.cfg.mol_graph_collator_cfg.name == 'cl':
            return -1
        else:
            raise ValueError(f"collator {self.cfg.mol_graph_collator_cfg.name} not supported.")
        

def train_only_dataset_to_dataloader(dataset_cls):
    class TrainOnlyDataLoader(BaseDataLoader):
        def __init__(self, cfg: DictConfig):
            super().__init__(cfg)
        
        def _get_dataset(self, split=None):
            dataset = dataset_cls(self.cfg)
            return dataset 
    return TrainOnlyDataLoader

MolGraphSSLDataLoader = train_only_dataset_to_dataloader(MolGraphMPPSSLDataset)
MolGraphSSLDataLoader = register_dataset('mol_graph_ssl', MolGraphSSLDatasetConfig)(MolGraphSSLDataLoader)

if __name__ == "__main__":
    from tqdm import tqdm 
    import torch 
    from vemol.dataset.mol_graph_config.mol_graph_form_config import MOL_GRAPH_FORM_DATACLASS_REGISTRY
    from vemol.dataset.mol_graph_config.mol_graph_collator_config import MOL_COLLATOR_DATACLASS_REGISTRY
    
    torch.manual_seed(1)
    cfg = MolGraphSSLDatasetConfig()
    cfg.data_parallel_size = 1
    cfg.validate = False 
    # cfg.num_workers = 8
    cfg.batch_size = 32
    cfg.downsize = -1
    cfg.seed = 1
    local_defaults = [{"mol_graph_form_cfg": "atom_graph"}, 
                  {"mol_graph_collator_cfg": "mae"},]
    cfg.name = 'chembl29'
    
    cfg.mol_graph_form_cfg = DictConfig(MOL_GRAPH_FORM_DATACLASS_REGISTRY[local_defaults[0]['mol_graph_form_cfg']])
    cfg.mol_graph_collator_cfg = DictConfig(MOL_COLLATOR_DATACLASS_REGISTRY[local_defaults[1]['mol_graph_collator_cfg']])
    
    cfg.mol_graph_collator_cfg.lmdb_path = '/mnt/nvme_share/wangjx/chembl29_pretrain_raw_graphs/lmdb'
    # cfg.use_cache_data = False
    dataloader = MolGraphSSLDataLoader(cfg)
    print(f"length of dataloader: {dataloader.num_batches_per_epoch}")
    for i, data in tqdm(enumerate(dataloader.get_dataloader('train'))): # 3 min
        print(data['atom_graph'].ndata['h'].shape)
        print(data['atom_graph'].batch_num_nodes()); break
        pass 
        # if i>100:
        #     break