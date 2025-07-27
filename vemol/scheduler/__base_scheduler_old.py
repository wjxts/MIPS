import os

from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import Optimizer
from builtins import str
from dataclasses import field, dataclass
from omegaconf import II, MISSING
from typing import Any, List, Optional, Union



from vemol.dataclass import BaseDataclass
from abc import ABC,abstractmethod 


@dataclass
class BaseSchedulerConfig(BaseDataclass):
    lr: float = 0.001
    # warmup: bool = False #'GradualWarmupScheduler'
    warmup_epochs: int = -1
    warmup_steps: int = -1
    multiplier: int = 1
    epochs: int = II("common.epochs")

class BaseScheduler(ABC):
    def __init__(self, cfg: BaseSchedulerConfig, optimizer: Optimizer):
        assert isinstance(optimizer, Optimizer), f"{optimizer} is not an Optimizer object!"
        self.cfg = cfg
        # self.scheduler = None
        self.optimizer = optimizer
        self.scheduler = self._build_scheduler()
        
    
    @abstractmethod
    def _build_scheduler(self): # -> Union[self, _LRScheduler]
        """
        子类可选的实现
        """
        return self
    
    def get_scheduler(self):
        return self.scheduler
    
    @abstractmethod  
    def state_dict(self):
        pass 
    
    @abstractmethod  
    def load_state_dict(self):
        pass
    
    def step_update(self, step=None):
        return self._step_update(step)

    @abstractmethod
    def _step_update(self, step=None):
        return None

    def epoch_update(self, epoch=None):
        return self._epoch_update(epoch)

    @abstractmethod
    def _epoch_update(self, epoch=None):
        return None
    

if __name__ == "__main__":
    pass 
    
    
            
        
    
        
    