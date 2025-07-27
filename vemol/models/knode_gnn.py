from dataclasses import dataclass, field
import math
from typing import List, Dict

import dgl 
from omegaconf import II, MISSING
import torch.nn as nn
import torch

# from vemol.chem_utils.featurizer import ATOM_FEATURE_DIM
from vemol.models import register_model
# from vemol.models.base_model import ModelConfig, BaseModel
from vemol.models.base_model import base_model_wrapper, cl_model_wrapper

from vemol.models.base_gnn import BaseGNNConfig, BaseGNN

from vemol.chem_utils.fingerprint.fingerprint import FP_DIM

@dataclass
class KnodeGNNConfig(BaseGNNConfig):
    model: str = "knode_gnn"
    kvec_names: List[str] = II("dataset.kvec_names")
    

class KnodeGNN(BaseGNN): 
    def __init__(self, cfg: KnodeGNNConfig):
        super().__init__(cfg)
        self.init_kvec_params()
        
    def init_kvec_params(self):
        self.kvec_names = self.cfg.kvec_names
        self.kvec_proj = nn.ModuleDict({name: nn.Linear(FP_DIM[name], self.cfg.d_model) for name in self.kvec_names})
        
    def get_init_node_feature(self, data):
        g = data['atom_graph']
        kvecs = data['kvecs']
        node_indicator = g.ndata['node_indicator']
        g.ndata['h'] = self.proj_in(g.ndata['h']) # 先都映射到d_model
        for i, name in enumerate(self.kvec_names, start=1):
            g.ndata['h'][node_indicator==i] = self.kvec_proj[name](kvecs[name])
        return g.ndata['h']


WrappedKnodeGNN = base_model_wrapper(KnodeGNN)
register_model("knode_gnn", dataclass=KnodeGNNConfig)(WrappedKnodeGNN)