import os
import torch
from builtins import str
from dataclasses import field
from omegaconf import II, MISSING

import torch.nn as nn
import torch
import torch.distributed as dist


from dataclasses import dataclass
from vemol.dataclass import BaseDataclass
from vemol.utils.distributed import get_model

import math 

@dataclass
class CriterionCfg(BaseDataclass):
    criterion: str = field(
        default=None, metadata={'help': "The name of criterion."}
    )
    device: int = II("common.device")
    module_loss: bool = False
    label_smoothing: float = 0.0
    data_parallel_size: int = II("common.data_parallel_size")
    valid_criterion: str = "acc"

    
class Criterion(nn.Module):
    def __init__(self, cfg: CriterionCfg):
        super(Criterion, self).__init__()
        self.cfg = cfg
        self.criterion = None
        
        self.loss = 0
        self.correct = 0
        self.total = 0

        self.mode = 'train'
        self.model = None

    def set_model(self, model):
        self.model = get_model(model)
    
    def module_loss(self):
        m_loss = 0
        for name, m in self.model.named_modules():
            if hasattr(m,'loss'):
                # print(name)
                m_loss += m.loss()
        return m_loss
        
    def get_criterion(self):
        return self.criterion
    
        
    def forward(self, predict, labels):
        result = self._forward(predict, labels)
        if self.cfg.module_loss and self.training:
            module_loss = self.module_loss()
            # print(f"module_loss: {module_loss}");exit()
            result['loss'] += module_loss
        # if "scale" in result.keys():
        #     result['loss'] *= result['scale']
        return result

    def _forward(self, predict, labels): # 在父类中不实现_forward比较规范?
        result = {}
        result['loss'] = self.criterion(predict, labels[1])
        result['predict'] = predict
        result['label'] = labels[1]
        result['sample_size'] = 1
        result['log'] = {}
        return result
        
    
    def start(self, mode='train'):
        self.loss = 0
        self.correct = 0
        self.total = 0
        self.mode = mode
        
    # def cal_ones(self, result):
    #     return self._cal_ones(result)
    
    def _cal_ones(self, result):
        return None
        
    def end(self, result):
        self._cal_ones(result)
        return self._end()
    
    def _end(self):
        return None
    
    
        
        

    
        
    