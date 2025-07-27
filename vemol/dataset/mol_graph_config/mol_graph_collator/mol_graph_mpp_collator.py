from dataclasses import dataclass, field
import logging

from omegaconf import II, MISSING
from pathlib import Path
from typing import List, Any, Literal, Callable, Type

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch 
import dgl

from vemol.dataset.mol_graph_config.mol_graph_collator import register_mol_graph_collator

from vemol.dataset.mol_graph_config.mol_graph_collator.base_collator import BaseCLSCollator

from vemol.dataset.mol_graph_config.mol_graph_collator.base_collator import (get_batch_kvecs, 
                                                                             batch_atom_graphs, 
                                                                             batch_fragment_graphs,
                                                                             batch_iterative_line_graphs
                                                                             )


class MolGraphMPPCLSCollator(BaseCLSCollator):
    
    def get_batch_graphs(self, graph_items):
        raise NotImplementedError 
    
    def batch_data_list(self, data_list):
        # data_list的每一项包含了一个字段的所有数据
        graph_items, labels, kvecs_list = data_list[0], data_list[1], data_list[2]
        data = self.get_batch_graphs(graph_items)
        if isinstance(labels[0], torch.Tensor): # 用于力场标签
            labels = torch.concatenate(labels, dim=0)
        else:
            labels = torch.FloatTensor(labels)
        batch_kvecs = get_batch_kvecs(kvecs_list)
        data['labels'] = labels
        data['kvecs'] = batch_kvecs
        return data 

@register_mol_graph_collator('dummy_graph_mpp')
@register_mol_graph_collator('polymer_atom_graph_mpp')
@register_mol_graph_collator('atom_graph_mpp')
class AtomGraphMPPCLSCollator(MolGraphMPPCLSCollator):
    
    def get_batch_graphs(self, graph_items):
        return batch_atom_graphs(graph_items)
