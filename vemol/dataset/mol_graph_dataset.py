from abc import ABC, abstractmethod 
from dataclasses import dataclass, field
from functools import partial
import logging
import pickle
from pathlib import Path
from typing import List, Any, Literal, Callable, Type

from joblib import Parallel, delayed
import lmdb
import numpy as np
from omegaconf import II, MISSING
import pandas as pd
from tqdm import tqdm
import torch 
import dgl
from torch.utils.data import Dataset, DataLoader

from vemol.dataset import register_dataset
from vemol.dataset.base_dataset import BaseDatasetConfig, BaseDataLoader

from vemol.chem_utils.dataset_path import BENCHMARK_NAME, BENCHMARK_BASE_PATH
from vemol.chem_utils.dataset_path import get_split_name, get_task_type
from vemol.chem_utils.dataset_path import SPLIT_TO_ID
from vemol.chem_utils.dataset_path import DATASET_CACHE_DIR, DATASET_TASKS



from vemol.chem_utils.featurizer.standard_featurizer import ATOM_FEATURE_DIM, BOND_FEATURE_DIM, ATOM_SPATIAL_FEATURE_DIM
from vemol.chem_utils.smiles2graph.base_graph import smiles2base_graph, smiles2dummy_graph

from vemol.chem_utils.fingerprint.fingerprint import FP_DIM, get_batch_fp

from vemol.dataset.mol_graph_config.mol_graph_collator import get_mol_graph_collator_cls

from vemol.chem_utils.smiles2graph.dove_graph import Smiles2DoveBaseGraph

from vemol.chem_utils.smiles2graph.knode_graph import add_knodes_to_complete_graph, add_knodes_to_base_graph
from vemol.chem_utils.smiles2graph.complete_graph import smiles2complete_graph, base_graph2complete_graph
from vemol.chem_utils.smiles2graph.polymer_base_graph import psmiles2polymer_base_graph
# from filelock import FileLock
from vemol.chem_utils.polymer_utils import augment_polymer, psmiles_star_sub
from vemol.chem_utils.utils import encode_smiles

logger = logging.getLogger(__name__)

local_defaults = [{"mol_graph_form_cfg": "atom_graph"}, 
                  {"mol_graph_collator_cfg": "mpp"},]
                                                                                                                                                                                                        
@dataclass
class BaseMolGraphDatasetConfig(BaseDatasetConfig): 
    # 用default参数告诉hydra那个变量在config的层次里, 来初始化当前层次的config (把当前节点当作局部根节点)
    # defaults: List[Any] = field(default_factory=lambda: local_defaults)
    defaults: Any = MISSING
    dataset: str = field( 
        default='mol_graph_base', metadata={'help': "The class of dataset."}
    )
    name: str = field(
        default='bbbp', metadata={'help': "The name of dataset."}
    )
    # split_type: str = field(
    #     default='scaffold', metadata={'help': "The split type of dataset."} # e.g., scaffold, random, or others
    # )
    # split_id: int = field(
    #     default=0, metadata={'help': "The split id of dataset."}
    # )
    scaffold_id: int = field(
        default=0, metadata={'help': "The scaffold id of dataset."}
    )
    
    ### graph operation config 
    
    # For Graph Transformer
    completion: bool = field(
        default=False, metadata={'help': "whether transform to complete graph (k-hop) for graph transformer."}
    )
    max_length: int = field(
        default=3, metadata={'help': "max length of the path in complete graph."}
    )
    
    # For Line GNN
    lining: bool = field(
        default=False, metadata={'help': "whether transform to line graph."}
    )
    
    # For incorporating knowledge features, e.g., ecfp, md, rdkfp
    kvec_names: List[str] = field(
        default_factory=lambda: []
    )  # 命令行输入: dataset.kvec_names=[ecfp,md,rdkfp], 注意逗号间不要加空格
    # kvec_names: Tuple[str,...] = ('ecfp', 'md') # 和上边的都行
    add_knodes: bool = field(
        default=False, metadata={'help': "whether add knodes to the graph."}
    )
    
    select_targets: List[str] = field(
        default_factory=lambda: [], metadata={'help': "The targets to select."}
    )
    # For regression tasks
    # target_mean: Any = None 
    # target_std: Any = None 
    target_mean: List[float] = field(
        default_factory=lambda: [], metadata={'help': "mean."}
    )
    target_std: List[float] = field(
        default_factory=lambda: [], metadata={'help': "std."}
    )
    debug: bool = False # if debug=True, load test split for train, to accelerate the training process
    use_cache_data: bool = True
    edge_input_dim: int = -1
    n_jobs: int = 1 # 预处理数据的进程数
    seed: int = II("common.seed")
    
    downsize: int = field(
        default=-1, metadata={'help': "The size of dataset. -1 means all. For debugging."}
    )
    
    mol_graph_form_cfg: Any = MISSING # 与输入图的形式有关的参数设置, atom_graph, fragment_graph
    mol_graph_collator_cfg: Any = MISSING # 预训练的参数设置
    # collator是由这两个cfg决定的
    
    
    
    
# 都能复用的部分
class BaseMolGraphDataset(Dataset):
    def __init__(self, cfg, split: Literal["train", 'valid', 'test']=None) -> None:
        # 感觉init要把逻辑写的清楚点，具体操作可以封装为成员函数
        # print(cfg);exit()
        self.cfg = cfg 
        self.split = split if not self.cfg.debug else 'test'

        self.init_graph_data()
        self.load_kvecs()
        self.set_collator() # 设定用于dataloader的batch数据收集器
        self.set_model_dim() # 设定模型输入输出维度

    # 写成成员函数方便在子类修改，而不用修改init函数
    
    # 数据存储的路径统一设计为benchmark_path/{dataset_name}/{dataset_name}.csv
    @property
    def dataset_name(self) -> str:
        # 数据集的名称
        return self.cfg.name 
    
    @property
    def benchmark_name(self) -> Path:
        # 数据集对应benchmark的名称
        return BENCHMARK_NAME[self.dataset_name]
    
    @property
    def benchmark_path(self) -> Path:
        # 数据集对应benchmark的路径(一个文件夹)
        return BENCHMARK_BASE_PATH[self.benchmark_name]
    
    @property 
    def dataset_folder(self) -> Path:
        return self.benchmark_path / self.dataset_name
    
    @property
    def dataset_file(self) -> Path:
        # 数据集的文件路径
        return self.dataset_folder / f"{self.dataset_name}.csv"
    
    @property
    def task_type(self) -> Literal["cls", "reg"]:
        # 数据集的类型
        return get_task_type(self.dataset_name)
    
    def __get_split_dict(self) -> np.ndarray:
        split_name = get_split_name(dataset=self.dataset_name, scaffold_id=self.cfg.scaffold_id)
        split_path = self.benchmark_path / self.dataset_name / "splits" / f"{split_name}.npy"
        split_dict = np.load(split_path, allow_pickle=True) # 嵌套数组[0][1][2]分别对应train, valid, test的存储idx的1D array
        return split_dict
    
    def get_use_idxs(self) -> np.ndarray:
        split_dict = self.__get_split_dict()
        use_idxs = split_dict[SPLIT_TO_ID[self.split]]
        # if self.cfg.debug:
            # 只用train的前100个数据
            # use_idxs = use_idxs[:100]
        return use_idxs
    
    @property
    def cache_folder_name(self):
        complete_str = f"complete_length{self.cfg.max_length}_" if self.cfg.completion else ""
        suffix = f'_order{self.cfg.mol_graph_form_cfg.order}' if self.cfg.mol_graph_form_cfg.name == 'fragment_graph' else ''
        if self.cfg.mol_graph_form_cfg.name == 'polymer_atom_graph' or self.cfg.mol_graph_form_cfg.name == 'polymer_fragment_graph':
            suffix += f'_{self.cfg.mol_graph_form_cfg.op}'
            if self.cfg.mol_graph_form_cfg.main_chain_embed:
                suffix += '_mc'
        return f"{complete_str}{self.cfg.mol_graph_form_cfg.custom_name}{suffix}"
    
    @property
    def cache_data_dir(self):
        path = DATASET_CACHE_DIR  / self.dataset_name / self.cache_folder_name
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @property
    def cache_dataset_path(self):
        # 不应该和scaffold有关, 存一个整体的数据!
        return self.cache_data_dir / "graph.pt"
        # return self.cache_data_dir / f"{self.split}_scaffold{self.cfg.scaffold_id}.pt"
    
    @property
    def kvec_folder(self):
        folder = self.benchmark_path / self.dataset_name / "descriptor"
        folder.mkdir(parents=True, exist_ok=True)
        return folder
    
    def kvec_path(self, kvec_name) -> str:
        return self.kvec_folder / f"{kvec_name}.npy"
        
    def load_kvec_on_disk(self, kvec_name) -> np.ndarray:
        path = self.kvec_path(kvec_name)
        assert path.exists(), f"{path} not exists. Please generate first."
        # logger.info(f"load {kvec_name} from {path}")
        print(f"load {kvec_name} from {path}")
        kvec = np.load(path, mmap_mode='r') # 处理大特征矩阵的方法
        return kvec 
    
    def generate_kvec(self, kvec_name) -> np.ndarray:
        logger.info(f"process {kvec_name} descriptor")
        if self.cfg.mol_graph_form_cfg.name == 'polymer_atom_graph':
            smiles_list = [psmiles_star_sub(smiles) for smiles in self.total_smiles_list]
        else:
            smiles_list = self.total_smiles_list
        kvec = get_batch_fp(smiles_list, fp_name=kvec_name, n_jobs=self.cfg.n_jobs)
        
        # remove nan value
        nan_count = np.isnan(kvec).sum()
        total_count = kvec.size # 统计总元素的数量
        nan_ratio = nan_count / total_count
        # logger.info(f"nan ratio of {kvec_name}={nan_ratio}")
        print(f"nan ratio of {kvec_name}={nan_ratio}")
        if nan_ratio > 0:
            kvec = np.nan_to_num(kvec, nan=0) # md feature似乎可能出现nan
        
        path = self.kvec_path(kvec_name)
        np.save(path, kvec) 
        print(f"save {kvec_name} to {path}")
        return kvec 
    
    def load_kvec_in_memory(self, kvec_name) -> np.ndarray:
        path = self.kvec_path(kvec_name)
        kvec = np.load(path)
        return kvec
    
    @property
    def in_memory_graph(self) -> bool:
        raise NotImplementedError
    
    @property
    def in_memory_kvec(self) -> bool:
        # 是否load kvec到内存
        raise NotImplementedError
    
    def load_kvec(self, kvec_name) -> np.ndarray:
        path = self.kvec_path(kvec_name)
        if self.in_memory_kvec:
            if path.exists():
                kvec = self.load_kvec_in_memory(kvec_name)
            else:
                kvec = self.generate_kvec(kvec_name)
        else:
            kvec = self.load_kvec_on_disk(kvec_name)
        return kvec 
    
    @property
    def edata_dim(self):
        raise NotImplementedError
    
    @property
    def add_knodes_to_graph(self):
        if self.cfg.completion:
            return add_knodes_to_complete_graph
        else:
            return add_knodes_to_base_graph
    
    def load_kvecs(self) -> None:
        self.kvec_data = {}
        for name in self.cfg.kvec_names:
            kvec = self.load_kvec(name)
            if self.in_memory_kvec:
                kvec = kvec[self.use_idxs] # 此处valid是有效的意思，不是validation set 
            # 在dataset文件夹里可以多进程测试一下mol2smiles是否成功, 然后就不需要valid_index了. 
            # 鲁棒一点加上好，因为rdkit不同版本是否成功有差异. 但目前固定了版本也没事. 
            self.kvec_data[name] = kvec
            
        if self.cfg.add_knodes and self.in_memory_graph:
            print("add knodes")
            self.graphs = Parallel(n_jobs=self.cfg.n_jobs)(delayed(self.add_knodes_to_graph)(graph, self.kvec_data, self.edata_dim) for graph in tqdm(self.graphs))
    
    
    def init_pretrain_data(self):
        if self.cfg.use_cache_data:
            self.init_on_disk_dataset()
        else:
            self.load_smiles()
    
    def init_on_disk_dataset(self):
        lmdb_path = self.cfg.mol_graph_collator_cfg.lmdb_path
        print(f"init {lmdb_path}")
        self.lmdb_env = lmdb.open(lmdb_path, readonly=True,
                                lock=False, readahead=False, meminit=False) # GPT说会有一点好处
    
    def init_in_memory_dataset(self):
        raise NotImplementedError
    
    def init_graph_data(self):
        raise NotImplementedError
    
    def get_data_item_on_disk(self, idx: int): # 不包括target
        with self.lmdb_env.begin() as txn:
            key = encode_smiles(self.total_smiles_list[idx])
            value = txn.get(key)
            data_item = pickle.loads(value)  
            # exit()
        return data_item 
    
    def get_data_item_in_memory(self, idx):
        pass 
    
    def get_data_item_on_the_fly(self, idx):
        pass 
    
    def __len__(self):
        raise NotImplementedError
    
    def __getitem__(self, idx):
        raise NotImplementedError
    
    @property
    def target_dim(self):
        raise NotImplementedError
        
    def set_model_dim(self):
        if self.cfg.input_dim<0: # 优先使用config里指定的
            self.cfg.input_dim = ATOM_FEATURE_DIM
        if self.cfg.output_dim<0:
            self.cfg.output_dim = self.target_dim
        if self.cfg.edge_input_dim<0:
            self.cfg.edge_input_dim = BOND_FEATURE_DIM
    
    @property
    def collator_cls(self):
        return get_mol_graph_collator_cls(self.cfg)
    
    def set_collator(self):
        self.collator = self.collator_cls(self.cfg)
    
    def load_smiles(self) -> None:
        df = pd.read_csv(self.dataset_file)
        self.total_smiles_list = df["smiles"].values.tolist()
        
    def load_smiles_and_targets(self) -> None:
        # 用于性质预测任务
        df = pd.read_csv(self.dataset_file)
        self.use_idxs = self.get_use_idxs()
        
        if self.cfg.downsize > 0:
            # 只用前N个数据
            print(f"downsizing {self.split} set: {self.cfg.downsize}")
            self.use_idxs = self.use_idxs[:self.cfg.downsize]
        
        self.total_smiles_list = df["smiles"].values.tolist() # 在后边kvec dataset处理fp特征时用到
        df = df.iloc[self.use_idxs]
        self.smiles_list = df["smiles"].values.tolist()
        df.drop("smiles", axis=1, inplace=True)
        if len(self.cfg.select_targets)>0:
            df = df[self.cfg.select_targets]
        self.targets = df.values.tolist()
        self.num_targets = len(self.targets[0])
        if len(self.cfg.select_targets)==0:
            assert self.num_targets == DATASET_TASKS[self.dataset_name], f"# of targets mismatch: {self.num_targets} != {DATASET_TASKS[self.dataset_name]}"
    
    @property
    def fragmentor(self):
        if not hasattr(self, '_fragmentor'):
            self._fragmentor = Smiles2DoveBaseGraph(self.cfg.mol_graph_form_cfg.vocab_path, 
                                                    order=self.cfg.mol_graph_form_cfg.order,
                                                    atom_graph_cache_lmdb_path=self.cfg.mol_graph_form_cfg.atom_graph_lmdb_path,
                                                    fragment_graph_cache_lmdb_path=self.cfg.mol_graph_form_cfg.fragment_graph_lmdb_path)
        return self._fragmentor
    
    @property
    def smiles2graph(self) -> Callable[[str], Any]:
        graph_form = self.cfg.mol_graph_form_cfg.name
        if graph_form == 'atom_graph':
            if self.cfg.completion:
                def composed_func(smiles):
                    base_graph = smiles2base_graph(smiles)
                    return base_graph2complete_graph(base_graph, max_length=self.cfg.max_length)
                return composed_func
            else:
                return smiles2base_graph
            
        elif graph_form == 'fragment_graph':
                 
            if self.cfg.completion:
                def composed_func(smiles):
                    item = self.fragmentor(smiles)
                    item['fragment_graph'] = base_graph2complete_graph(item['fragment_graph'], max_length=self.cfg.max_length)
                    return item
                return composed_func
            else:
                return self.fragmentor
        elif graph_form == 'polymer_atom_graph':
            op = self.cfg.mol_graph_form_cfg.op
            main_chain_embed = self.cfg.mol_graph_form_cfg.main_chain_embed
            
            if self.cfg.completion:
                def composed_func(smiles):
                    base_graph = psmiles2polymer_base_graph(smiles, op=op, main_chain_embed=main_chain_embed)
                    return base_graph2complete_graph(base_graph, max_length=self.cfg.max_length)
                return composed_func
            else:
                return partial(psmiles2polymer_base_graph, op=op, main_chain_embed=main_chain_embed)
        
        elif graph_form == 'polymer_fragment_graph':
            op = self.cfg.mol_graph_form_cfg.op
            main_chain_embed = self.cfg.mol_graph_form_cfg.main_chain_embed
            def func(smiles):
                data = self.fragmentor.smiles2dove_polymer_base_graph(smiles, op, main_chain_embed)
                if self.cfg.completion:
                    data['atom_graph'] = base_graph2complete_graph(data['atom_graph'], max_length=self.cfg.max_length)
                return data 
            return func
        
        elif graph_form == 'dummy_graph':
            return smiles2dummy_graph
        else:
            raise ValueError(f"graph form {graph_form} not supported.")
    
        
    def load_cached_dataset(self, path: Path) -> List:
        logger.info(f"load cached dataset from {path}")
        with open(self.cache_dataset_path, "rb") as f:
            data_items = torch.load(f, map_location="cpu", weights_only=False)
        return data_items
    
    def save_cached_dataset(self, data_items: List, path: Path) -> None:
        logger.info(f"save cached dataset to {path}")
        torch.save(data_items, path)
        
    # @property
    # def smiles2graph(self) -> Callable[[str], Any]:
    #     # 用于输入smiles的数据集, 不适用于3D数据
    #     raise NotImplementedError 
    
    def process_smiles_list(self, smiles_list):
        logger.info(f"parallel convert smiles to graph, n_jobs={self.cfg.n_jobs}")
        graphs = Parallel(n_jobs=self.cfg.n_jobs)(delayed(self.smiles2graph)(s) for s in tqdm(smiles_list))
        return graphs 
    
    def check_graph_valid_index(self):
        if not hasattr(self, 'graphs'):
            self.valid_index_list = [i for i in range(len(self.targets))]
            return 
        self.valid_index_list = [i for i in range(len(self.graphs)) if self.graphs[i] is not None]
        self.graphs = [self.graphs[i] for i in self.valid_index_list]
        self.targets = [self.targets[i] for i in self.valid_index_list]
        self.kvec_data = {name: [kvec[i] for i in self.valid_index_list] for name, kvec in self.kvec_data.items()}

    def smiles2graphs(self):
        # 操作就是将smiles转换为图，最后再提取能成功转化的indices
        logger.info("smiles2graphs")
        if self.cache_dataset_path.exists() and self.cfg.use_cache_data:
            total_graphs = self.load_cached_dataset(self.cache_dataset_path)
            self.graphs = [total_graphs[i] for i in self.use_idxs]
        else:
            if self.cfg.use_cache_data:
                total_graphs = self.process_smiles_list(self.total_smiles_list)
                self.save_cached_dataset(total_graphs, self.cache_dataset_path)
                self.graphs = [total_graphs[i] for i in self.use_idxs]
            else:
                self.graphs = self.process_smiles_list(self.smiles_list)
            
    def __del__(self):
        # 关闭LMDB环境
        if hasattr(self, 'lmdb_env'):
            self.lmdb_env.close()
# 回归任务数据集相对分类任务要多一些额外的处理, 才用继承的方式编写代码
# 下游子类的回归任务，同时继承它对应的分类任务和这个回归任务基类


if __name__ == "__main__":
    pass 