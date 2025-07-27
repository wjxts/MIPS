from abc import ABC, abstractmethod 
from dataclasses import dataclass, field
import logging

from omegaconf import II, MISSING, DictConfig
from pathlib import Path
from typing import List, Any, Literal, Callable, Type

from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch 
import dgl
from torch.utils.data import Dataset, DataLoader

from vemol.dataset import register_dataset
from vemol.dataset.base_dataset import BaseDatasetConfig, BaseDataLoader

from vemol.chem_utils.dataset_path import BENCHMARK_NAME, BENCHMARK_BASE_PATH
from vemol.chem_utils.dataset_path import get_split_name, get_task_type

from vemol.chem_utils.dataset_path import DATASET_CACHE_DIR, DATASET_TASKS



from vemol.chem_utils.featurizer.standard_featurizer import ATOM_FEATURE_DIM, BOND_FEATURE_DIM
from vemol.chem_utils.smiles2graph.base_graph import smiles2base_graph
from vemol.chem_utils.smiles2graph.complete_graph import smiles2complete_graph, base_graph2complete_graph


from vemol.chem_utils.fingerprint.fingerprint import FP_DIM, get_batch_fp


from vemol.dataset.mol_graph_config.mol_graph_collator import get_mol_graph_collator_cls
from vemol.dataset.mol_graph_config.mol_graph_collator.base_collator import cls_to_reg_collator

from vemol.dataset.mol_graph_dataset import BaseMolGraphDatasetConfig, BaseMolGraphDataset
# from filelock import FileLock

from vemol.chem_utils.smiles2graph.knode_graph import add_knodes_to_complete_graph, add_knodes_to_base_graph

from vemol.chem_utils.smiles2graph.dove_graph import Smiles2DoveBaseGraph

logger = logging.getLogger(__name__)


local_defaults = [{"mol_graph_form_cfg": "atom_graph"}, 
                  {"mol_graph_collator_cfg": "mpp"},]

@dataclass
class MolGraphMPPDatasetConfig(BaseMolGraphDatasetConfig):
    # 用default参数告诉hydra那个变量在config的层次里, 来初始化当前层次的config (把当前节点当作局部根节点)
    defaults: List[Any] = field(default_factory=lambda: local_defaults)
    dataset: str = field(
        default='mol_graph_mpp', metadata={'help': "The class of dataset."}
    )
    


class MolGraphMPPCLSDataset(BaseMolGraphDataset):
    def __init__(self, cfg, split: Literal["train", 'valid', 'test']) -> None:
        # 感觉init要把逻辑写的清楚点，具体操作可以封装为成员函数
        super().__init__(cfg, split)
        self.check_graph_valid_index()
        
    def init_graph_data(self):
        # print("init")
        self.load_smiles_and_targets()
        if self.cfg.mol_graph_collator_cfg.lmdb_path:
            # print("init on disk dataset")
            self.init_on_disk_dataset()
        else:
            self.smiles2graphs()
        
        
    @property
    def in_memory_graph(self) -> bool:
        return True
    
    @property
    def in_memory_kvec(self) -> bool:
        # 是否load kvec到内存
        return True
    
    @property
    def target_dim(self):
        return self.num_targets
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        if self.cfg.mol_graph_collator_cfg.lmdb_path:
            graph = self.get_data_item_on_disk(self.use_idxs[idx])
            if self.cfg.completion:
                graph = base_graph2complete_graph(graph, max_length=self.cfg.max_length)
            target = self.targets[idx]
        else:
            graph, target = self.graphs[idx], self.targets[idx]
        kvecs = {name: self.kvec_data[name][idx] for name in self.cfg.kvec_names}
        return graph, target, kvecs
    
    @property
    def edata_dim(self):
        if self.cfg.completion:
            return {'distance':1, 'paths': self.cfg.max_length}
        else:
            return {'e': BOND_FEATURE_DIM}
        
    

# 回归任务数据集相对分类任务要多一些额外的处理, 才用继承的方式编写代码
# 下游子类的回归任务，同时继承它对应的分类任务和这个回归任务基类

def cls_to_reg_dataset(cls_dataset_class):
    
    class MPPREGDataset(cls_dataset_class):
        def __init__(self, cfg, split: Literal["train", 'valid', 'test']) -> None:
            super().__init__(cfg, split)
            self.set_mean_and_std(cfg, split)
        
        def __getitem__(self, idx):
            items = super().__getitem__(idx)
            return (*items, self.cfg.target_mean, self.cfg.target_std)
            
        def set_collator(self):
            self.collator = cls_to_reg_collator(super().collator_cls)()
        
        def set_mean_and_std(self, cfg, split):
            logger.info("normalize target for regression task")
            if split == 'train' or (split == 'test' and cfg.debug):
                self.__set_mean_and_std(cfg)
                
        def __set_mean_and_std(self, cfg):
            mean = np.nanmean(np.array(self.targets), axis=0)
            std = np.nanstd(np.array(self.targets), axis=0)
            # print(cfg.target_mean, cfg.target_std)
            if len(cfg.target_mean) == 0:
                cfg.target_mean = mean.tolist() # used in main.py
            if len(cfg.target_std) == 0:
                cfg.target_std = std.tolist()
            logger.info(f"mean={mean}, std={std}")
            logger.info(f"target mean={cfg.target_mean}, std={cfg.target_std}")
            
            # exit()
            
    return MPPREGDataset
      
def mpp_cls_dataset_to_dataloader(dataset_cls):
    class MPPDataLoader(BaseDataLoader):
        def __init__(self, cfg: BaseMolGraphDatasetConfig):
            self.task_type = get_task_type(cfg.name)
            # print(self.task_type)
            logger.info(f"dataset:{cfg.name}, task type: {self.task_type}")
            super().__init__(cfg)
        
        @property
        def cls_dataset_class(self):
            return dataset_cls
        
        @property
        def reg_dataset_class(self):
            return cls_to_reg_dataset(self.cls_dataset_class)
        
        def _get_dataset(self, split):
            if self.task_type == 'cls':
                dataset = self.cls_dataset_class(cfg=self.cfg, split=split)
            else:
                dataset = self.reg_dataset_class(cfg=self.cfg, split=split)
            return dataset 
    return MPPDataLoader


MolGraphMPPDataLoader = mpp_cls_dataset_to_dataloader(MolGraphMPPCLSDataset)
MolGraphMPPDataLoader = register_dataset('mol_graph_mpp', MolGraphMPPDatasetConfig)(MolGraphMPPDataLoader)


if __name__ == "__main__":
    # 这里logger在运行main()时可以打印info, 但是这里local运行不能打印出信息
    # 这里的logger和外边的logger不一样
    # print("main1", logger) # main <Logger __main__ (INFO)>
    from vemol.dataset.mol_graph_config.mol_graph_form_config import MOL_GRAPH_FORM_DATACLASS_REGISTRY
    from vemol.dataset.mol_graph_config.mol_graph_collator_config import MOL_COLLATOR_DATACLASS_REGISTRY
    torch.manual_seed(0)
    logger.info("main")
    cfg = MolGraphMPPDatasetConfig() 

    local_defaults = [{"mol_graph_form_cfg": "atom_graph"}, 
                  {"mol_graph_collator_cfg": "mpp"},]
    # local_defaults = [{"mol_graph_form_cfg": "fragment_graph"}, 
    #               {"mol_graph_collator_cfg": "mpp"},]
    
    cfg.mol_graph_form_cfg = DictConfig(MOL_GRAPH_FORM_DATACLASS_REGISTRY[local_defaults[0]['mol_graph_form_cfg']])
    cfg.mol_graph_collator_cfg = DictConfig(MOL_COLLATOR_DATACLASS_REGISTRY[local_defaults[1]['mol_graph_collator_cfg']])
    cfg.data_parallel_size = 1
    # cfg.completion = True
    # cfg.use_cache_data = True
    cfg.name = 'bace'
    # cfg.name = 'pcba'
    # cfg.name = 'freesolv'
    cfg.n_jobs = 8
    # cfg.kvec_names = ['md', 'rdkfp']
    # cfg.add_knodes = True
    task_type = get_task_type(cfg.name)
    dataloader = MolGraphMPPDataLoader(cfg)
    for data in dataloader.get_dataloader('train'):
        print(data['atom_graph'].batch_num_nodes())
        if task_type == 'cls':
            print(data['labels'].shape)
        else:
            print(data['labels']['mean'])
        break
    # for data in dataloader.get_dataloader('valid'):
    #     print(data['g'].batch_num_nodes(), data['labels'].shape)
    #     break
    # for data in dataloader.get_dataloader('test'):
    #     print(data['g'].batch_num_nodes(), data['labels'].shape)
    #     break
    # print("done")
    
    # print("main2", logger) # main <Logger __main__ (INFO)>