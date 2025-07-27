from dataclasses import dataclass, field
import math
from typing import List, Dict

import dgl 
from omegaconf import II, MISSING
from omegaconf import DictConfig
import torch.nn as nn
import torch

# from vemol.chem_utils.featurizer import ATOM_FEATURE_DIM
from vemol.models import register_model
from vemol.models.base_model import ModelConfig, BaseModel
from vemol.models.base_model import base_model_wrapper, cl_model_wrapper

from vemol.modules.gnn_layer import GNN_LAYERS



@dataclass
class BaseGNNConfig(ModelConfig):
    model: str = "base_gnn"
    gnn_layer: str = 'gin'
    input_dim: int = -1 # 137
    output_dim: int = -1 # 
    d_model: int = 256
    num_layers: int = 2
    readout: str = 'mean'
    residual: bool = False
    pre_norm: bool = False
    norm_type: str = 'none'
    predictor_bias: bool = True
    predictor_norm: str = 'none'



# @register_model("base_gnn", dataclass=BaseGNNConfig)
# @base_model_wrapper
class BaseGNN(nn.Module): # 改成Base
    def __init__(self, cfg: BaseGNNConfig):
        super().__init__()
        # print(cfg)
        self.cfg = cfg
        self.proj_in = nn.Linear(cfg.input_dim, cfg.d_model, bias=False)
        # layer_cls = GNN_LAYERS[args.layer_name]
        layer_cls = GNN_LAYERS[cfg.gnn_layer]
        self.gnns = nn.ModuleList([layer_cls(cfg.d_model, cfg.d_model) for _ in range(cfg.num_layers)])
        self.readout = cfg.readout
        if cfg.output_dim<0:
            cfg.output_dim = cfg.d_model
        self.predictor = nn.Linear(cfg.d_model, cfg.output_dim, bias=cfg.predictor_bias) 
        # self.predictor = nn.Sequential(nn.LayerNorm(cfg.d_model), nn.Linear(cfg.d_model, cfg.output_dim, bias=False))
        # print(self.predictor);exit()
        
    def get_init_node_feature(self, data):
        g = data['atom_graph']
        return self.proj_in(g.ndata['h'])
        
    def forward(self, data: Dict):
        g = data['atom_graph']
        x = self.get_init_node_feature(data)
        for gnn in self.gnns:
            if self.cfg.residual:
                res = x
            x = gnn(g, x)
            if self.cfg.residual:
                x = x + res
        g.ndata['_pool_h'] = x
        if self.readout != 'none':
            x = dgl.readout_nodes(g, '_pool_h', op=self.readout)
        # feature_norm = x.norm(dim=-1)
        # print(x.shape, feature_norm**2/3/256, sep='\n') # 平均norm=300
        
        x = self.predictor(x)
        # print(self.predictor.bias) # 0.0083, 很小; 或者直接去掉
        # print(self.predictor.weight.shape, 1/(self.predictor.weight**2).mean())
        # print(x.shape, x.reshape(-1), sep='\n') # freesolv: 全是负的, 有点异常? 
        # print(x.norm(dim=-1))
        # print((x.norm(dim=-1)**2).mean());exit()   # 预训练之前是0.0008, 预训练之后是256, 加一个layernorm可以有效避免这个问题
        # x = x/16
        # x = x*math.sqrt(3)
        # print((x.norm(dim=-1)**2).mean());exit()
        # print(x.shape);exit()
        return x

# 每次写这个wrap有点丑, 可以写一个装饰器来自动wrap
# 但同一个model会被不同的包装，原始包装和CL包装

# @register_model("base_gnn", dataclass=BaseGNNConfig)
# class WrappedBaseGNN(BaseModel):
#     def __init__(self, cfg: BaseGNNConfig):
#         super().__init__(cfg)
#         self.model = BaseGNN(cfg)

WrappedBaseGNN = base_model_wrapper(BaseGNN)
register_model("base_gnn", dataclass=BaseGNNConfig)(WrappedBaseGNN)



@dataclass
class CLGNNConfig(BaseGNNConfig):
    model: str = "cl_gnn"
    
# @register_model("cl_gnn", dataclass=CLGNNConfig)
# class WrappedCLGNN(BaseModel):
#     def __init__(self, cfg: CLGNNConfig):
#         super().__init__(cfg)
#         self.model = CLWrapper(BaseGNN(cfg))

WrappedCLGNN = base_model_wrapper(cl_model_wrapper(BaseGNN))
register_model("cl_gnn", dataclass=CLGNNConfig)(WrappedCLGNN)

