import numpy as np
from dataclasses import dataclass, field
from typing import List

import torch.optim

from vemol.optimizer import register_optimizer
from vemol.optimizer.base_optimizer import OptimizerConfig, Optimizer
from hydra.core.config_store import ConfigStore

@dataclass
class AdamConfig(OptimizerConfig):
    # (self.optimizer, self.cfg.lr_step, self.cfg.lr_gamma)
    
    optimizer: str = field(
        default='adam', metadata={"help": "the optimizer method to train network)"}
    )
    
    lr: float = field(
        default=1e-3, metadata={'help': "The initial learning rate (default: 0.001)."}
    )
    beta1: float = field(
        default=0.9, metadata={'help': "Adam beta1."}
    )
    beta2: float = field(
        default=0.98, metadata={'help': "Adam beta2."}
    )
    eps: float = field(
        default=1e-8, metadata={'help': "Adam eps."}
    )
    weight_decay: float = field(
        default=0, metadata={"help": "weight decay (default: 0)."}
    )

@register_optimizer('adam', AdamConfig)
class AdamOptimizer(Optimizer):
    def __init__(self, cfg: AdamConfig):
        super(AdamOptimizer, self).__init__(cfg)
        
    def _build_optimizer(self, model):
        """
        build adam                                                 
        """
        params = self._get_model_grouped_parameters(model)
        self.optimizer = torch.optim.Adam(
            params, lr=self.cfg.lr, betas=(self.cfg.beta1, self.cfg.beta2), weight_decay=self.cfg.weight_decay
        )
        return self.optimizer

@dataclass
class AdamWConfig(AdamConfig):    
    optimizer: str = field(
        default='adamw', metadata={"help": "the optimizer method to train network)"}
    )
    
@register_optimizer('adamw', AdamWConfig)
class AdamWOptimizer(Optimizer):
    def __init__(self, cfg: AdamWConfig):
        super(AdamWOptimizer, self).__init__(cfg)
        
    def _build_optimizer(self, model):
        """
        build adamw                                                 
        """
        params = self._get_model_grouped_parameters(model)
        self.optimizer = torch.optim.AdamW(
            params, lr=self.cfg.lr, betas=(self.cfg.beta1, self.cfg.beta2), weight_decay=self.cfg.weight_decay
        )
        return self.optimizer
     
if __name__ == '__main__':
    cs = ConfigStore.instance()
    node = AdamConfig()
    name = 'adam'
    node._name = name
    cs.store(name=name, group="optimizer", node=node)
    print(node)
    optimizer = AdamOptimizer(node)
    # model = torchvision.models.resnet18(pretrained=False)
    # optimizer.set_model(model)
    # print(optimizer.get_optimizer())