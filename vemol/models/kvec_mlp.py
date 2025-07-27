
from dataclasses import dataclass
from omegaconf import II, MISSING
from typing import List 

import torch.nn as nn 
import torch

from vemol.models import register_model
from vemol.models.base_model import ModelConfig, BaseModel
from vemol.models.base_model import base_model_wrapper

from vemol.models.mlp import MLPConfig
from vemol.modules.mlp import MLPModule

from vemol.chem_utils.fingerprint.fingerprint import FP_DIM

@dataclass
class KvecMLPConfig(MLPConfig):
    model: str = "kvec_mlp"
    d_model: int = 512
    num_layers: int = 1
    kvec_names: List[str] = II("dataset.kvec_names")

class KvecMLP(nn.Module):
    # 没有用到input_dim
    def __init__(self, cfg: MLPConfig):
        super().__init__()
        self.cfg = cfg
        self.kvec_names = cfg.kvec_names
        self.kvec_proj = nn.ModuleDict({name: nn.Linear(FP_DIM[name], cfg.d_model) for name in self.kvec_names})
        self.mlp = MLPModule(d_in=cfg.d_model, 
                             d_out=cfg.output_dim, 
                             d_hidden=cfg.d_model, 
                             n_layers=cfg.num_layers, 
                             activation=nn.ReLU(), 
                             dropout=0.0)
        
    def forward(self, samples):
        kvecs = samples['kvecs']
        x = 0
        for name in self.kvec_names:
            x = x + self.kvec_proj[name](kvecs[name])  
        y = self.mlp(x)
        return y 


WrappedKvecMLP = base_model_wrapper(KvecMLP)
register_model("kvec_mlp", dataclass=KvecMLPConfig)(WrappedKvecMLP)


if __name__ == '__main__':
    cfg = KvecMLPConfig()
    cfg.input_dim = 10
    cfg.output_dim = 1
    kvec_names = ['ecfp', 'md']
    cfg.kvec_names = kvec_names
    model = WrappedKvecMLP(cfg)
    
    kvecs = {name: torch.randn(3, FP_DIM[name]) for name in kvec_names}
    samples = {'kvecs': kvecs}
    y = model(samples)
    print(y)