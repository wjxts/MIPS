from abc import ABC, abstractmethod 
from typing import Dict, List, Type

import dgl
import numpy as np 
import torch

class BaseCLSCollator(ABC):
    def __init__(self, *args, **kwargs) -> None:
        pass 
    
    @abstractmethod
    def batch_data_list(self, data_list):
        raise NotImplementedError("batch_data_list method is not implemented")  
    
    def __call__(self, item_list):
        data_list = list(zip(*item_list))
        data = self.batch_data_list(data_list)
        return data 


class BaseREGCollator:
    def __init__(self, *args, **kwargs) -> None:
        pass
    
    def __call__(self, item_list):
        data_list_with_stat = list(zip(*item_list))
        data_list = data_list_with_stat[:-2]
        means, stds = data_list_with_stat[-2], data_list_with_stat[-1]
        mean = means[0]
        std = stds[0]
        mean = np.array(mean) 
        std = np.array(std)
        
        data = self.batch_data_list(data_list) # should inherit from CLS Collator
        labels = {"labels": data["labels"], "mean":mean, "std":std}
        data['labels'] = labels
        return data

# 注意继承顺序, 需要REGCollator的__call__函数, 和CLSCollator的batch_data_list函数
# class BaseMolGraphMPPREGCollator(BaseREGCollator, BaseMolGraphMPPCLSCollator):
#     def __init__(self) -> None:
#         pass 

def cls_to_reg_collator(cls_collator_class: Type[BaseCLSCollator]) -> Type[BaseREGCollator]:
    class REGCollator(BaseREGCollator, cls_collator_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    return REGCollator


def get_batch_kvecs(kvecs_list: List[Dict[str, np.ndarray]]) -> Dict[str, torch.Tensor]:
    batch_kvecs = {}
    for name in kvecs_list[0]:
        batch_kvec = np.array([kvecs[name] for kvecs in kvecs_list]).astype(np.float32)
        batch_kvecs[name] = torch.from_numpy(batch_kvec)
    return batch_kvecs


def batch_atom_graphs(items: List[dgl.DGLGraph]):
    batch_graph = dgl.batch(items)
    data = {'atom_graph': batch_graph}
    return data

def value_shift_by_num_nodes(id_list, bg):
    # 将第i个图对应的数据值便宜 V[0]+...+V[i-1], V为节点数
    # id_list: a list of 1-D torch.LongTensor
    # 获取每个图的节点数
    num_nodes_per_graph = bg.batch_num_nodes()
    accum_num_nodes = torch.cat([torch.tensor([0], device=num_nodes_per_graph.device), 
                                    num_nodes_per_graph.cumsum(0)[:-1]])
    assert len(id_list) == len(accum_num_nodes), "id_list and accum_num_nodes should have the same length"
    # print(num_nodes_per_fragment_graph.cumsum(0))
    # 为每个图创建一个与节点数相同长度的tensor，填充值为图的
    shift_id_list = [id_list[i]+accum_num_nodes[i] for i in range(len(id_list))]
    # 拼接所有的tensor得到所有节点的batch_id
    batch_shift_id = torch.cat(shift_id_list)
    return batch_shift_id

def batch_fragment_graphs(items: List[Dict]):
    batch_atom_graph = dgl.batch([item['atom_graph'] for item in items])
    batch_fragment_graph = dgl.batch([item['fragment_graph'] for item in items])
    
    
    group_atom_idxs_1d_list = [item['group_atom_idxs_1d'] for item in items]
    batch_group_atom_idxs_1d = value_shift_by_num_nodes(group_atom_idxs_1d_list, batch_atom_graph)
    
    macro_node_scatter_idxs_list = [item['macro_node_scatter_idxs'] for item in items]
    batch_macro_node_scatter_idxs = value_shift_by_num_nodes(macro_node_scatter_idxs_list, batch_fragment_graph)
    
    data = {
            'atom_graph': batch_atom_graph,
            'fragment_graph': batch_fragment_graph, 
            'group_atom_idxs_1d': batch_group_atom_idxs_1d, 
            'macro_node_scatter_idxs': batch_macro_node_scatter_idxs,
        }
    return data 
   



if __name__ == "__main__":
    pass 