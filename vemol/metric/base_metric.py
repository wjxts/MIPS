import os
import torch
from builtins import str
from dataclasses import field
from omegaconf import II, MISSING

import torch.nn as nn
import torch


from dataclasses import dataclass
from vemol.dataclass import BaseDataclass

from abc import ABC,abstractmethod 
from typing import Dict 

@dataclass
class MetricCfg(BaseDataclass):
    metric: str = field(
        default=None, metadata={'help': "The name of metric."}
    )
    
    
class Metric(nn.Module, ABC):
    def __init__(self, cfg: MetricCfg):
        super(Metric, self).__init__()
        self.cfg = cfg
        self.metric = None
    
    def forward(self, predicts, labels):
        return self._forward(predicts, labels)
    
    @abstractmethod
    def _forward(self, predicts, labels) -> Dict[str, float]:
        # return the metric score: e.g., {'acc': 0.9, 'f1': 0.8}
        pass
    
        
        

    
        
    