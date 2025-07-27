
from dataclasses import dataclass
from omegaconf import II, MISSING

import torch.nn as nn 
import torch

from vemol.models import register_model
from vemol.models.base_model import ModelConfig, BaseModel
from vemol.modules.mlp import MLPModule
from vemol.modules.observer import Observer

@dataclass
class MLPConfig(ModelConfig):
    model: str = "mlp"
    input_dim: int = -1 
    output_dim: int = -1 
    d_model: int = 256
    num_layers: int = 1


class MLP(nn.Module):
    def __init__(self, cfg: MLPConfig):
        super().__init__()
        self.cfg = cfg

        self.mlp = MLPModule(d_in=cfg.input_dim, 
                             d_out=cfg.output_dim, 
                             d_hidden=cfg.d_model, 
                             n_layers=cfg.num_layers, 
                             activation=nn.ReLU(), 
                             dropout=0.0)
        # self.observer = Observer(name='Observer')
        self.observer = Observer()
        self.norm = nn.BatchNorm1d(cfg.input_dim)
        
    def forward(self, samples):
        x = samples['x']
        x.requires_grad = True
        x = self.norm(x)
        x = self.observer(x)
        y = self.mlp(x)
        return y 

@register_model("mlp", dataclass=MLPConfig)
class WrappedMLP(BaseModel):
    def __init__(self, cfg: MLPConfig):
        super().__init__(cfg)
        self.model = MLP(cfg)
    
    def valid_step(self, samples):
        y = self.model(samples)
        predict_id = torch.argmax(y, dim=-1, keepdim=True)
        return predict_id

if __name__ == "__main__":
    node = MLPConfig()
    node.input_dim = 10
    node.output_dim = 4
    model = WrappedMLP(node)
    
    x = torch.randn(3, 10)
    samples = {'x': x}
    print(model.train_step(samples))
    print(model.valid_step(samples))
    