import os

from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import Optimizer
from builtins import str
from dataclasses import field, dataclass
from omegaconf import II, MISSING
from typing import Any, List, Optional, Union



from vemol.dataclass import BaseDataclass
from abc import ABC,abstractmethod 

PER_STEP_MODE = "per_step"
PER_EPOCH_MODE = "per_epoch"

@dataclass
class BaseSchedulerConfig(BaseDataclass):
    scheduler: str = field(
        default='', metadata={"help": "the name of scheduler)"}
    )
    lr: float = 1e-3
    update_type: str = PER_STEP_MODE # per_step or per_epoch
    support_warmup_wrapper: bool = True
    warmup_epochs: int = -1
    warmup_steps: int = -1
    multiplier: int = 1
    epochs: int = II("common.epochs")
    num_batches_per_epoch: int = -1 # 由dataloader决定, 在Trainer的init里初始化

# 目前默认都不使用内置的scheduler
class BaseScheduler(ABC):
    def __init__(self, cfg: BaseSchedulerConfig, optimizer: Optimizer):
        assert isinstance(optimizer, Optimizer), f"{optimizer} is not an Optimizer object!"
        self.cfg = cfg
        # self.scheduler = None
        self.optimizer = optimizer
        # self.scheduler = self._build_scheduler()
        self.lr = -1
        # if not isinstance(self.scheduler, _LRScheduler):
        #     self.step_update(step=1) # 是否调用取决于是否需要在初始化时更新lr, pytorch内置的scheduler在初始化时已经更新了一步
        #     self.epoch_update(epoch=1)
        self.step_update(step=1)
        self.epoch_update(epoch=1)
    
    # @abstractmethod
    # def _build_scheduler(self): # -> Union[self, _LRScheduler]:
    #     """
    #     子类可选的实现
    #     """
    #     return self
    
    # def get_scheduler(self):
    #     return self.scheduler
    
    def set_lr(self, lr):
        self.lr = lr # 先set才能get
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            
    def get_lr(self) -> float:
        if self.lr < 0:
            raise ValueError(f"lr={self.lr} is not set yet!")
        return self.lr 
    
    def step_update(self, step: int):
        if self.cfg.update_type == PER_STEP_MODE:
            lr = self.get_step_lr(step)
            assert lr is not None, "lr is not set in step update!"
            self.set_lr(lr)
    
    def epoch_update(self, epoch: int):
        if self.cfg.update_type == PER_EPOCH_MODE:
            lr = self.get_epoch_lr(epoch)
            assert lr is not None, "lr is not set in epoch update!"
            self.set_lr(lr)

    @abstractmethod
    def get_step_lr(self, step: int) -> Optional[float]:
        raise NotImplementedError
    
    @abstractmethod
    def get_epoch_lr(self, epoch: int) -> Optional[float]:
        raise NotImplementedError
    
    # @abstractmethod  
    def state_dict(self):
        return {"cfg": self.cfg}
    
    # @abstractmethod  
    def load_state_dict(self, state_dict):
        self.cfg = state_dict["cfg"]


if __name__ == "__main__":
    pass 
    
    
            
        
    
        
    