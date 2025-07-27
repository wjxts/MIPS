import numpy as np
from dataclasses import dataclass, field


from torch.optim.lr_scheduler import CosineAnnealingLR
from omegaconf import II, MISSING
from hydra.core.config_store import ConfigStore

from vemol.scheduler import register_scheduler
from vemol.scheduler.base_scheduler import BaseSchedulerConfig, BaseScheduler
from vemol.scheduler.base_scheduler import PER_EPOCH_MODE
import math 

@dataclass
class COSSchedulerConfig(BaseSchedulerConfig):
    scheduler: str = field(
        default='cos', metadata={"help": "the scheduler method to modify lr)"}
    )
    update_type: str = PER_EPOCH_MODE
    T_max: int = II("common.epochs")
    eta_min: float = 1e-5
    

@register_scheduler('cos', COSSchedulerConfig)
class COSScheduler(BaseScheduler):
    def __init__(self, cfg: COSSchedulerConfig, optimizer=None):
        super(COSScheduler, self).__init__(cfg, optimizer)
        self.T_max = cfg.T_max
        self.eta_min = cfg.eta_min
        self.eta_max = cfg.lr 
        assert cfg.update_type == PER_EPOCH_MODE, f"update_type={cfg.update_type} is not supported for cosine scheduler!"
        
    # def _build_scheduler(self): # 可以不写
    #     return self
    # def _build_scheduler(self):
    #     """
    #     build cos
    #     """
    #     base_scheduler = CosineAnnealingLR(self.optimizer, T_max=self.cfg.T_max, eta_min=self.cfg.eta_min)
    #     return base_scheduler
    
    def get_step_lr(self, step: int) -> float:
        return None
    
    def get_epoch_lr(self, epoch: int) -> float:
        if self.cfg.warmup_epochs>0:
            epoch = epoch - self.cfg.warmup_epochs
        else:
            # 因为这个框架epoch从1开始, 所以计算lr时先减1
            epoch = epoch - 1 
        lr = self.eta_min + (self.eta_max - self.eta_min) * (1 + math.cos(epoch * math.pi / self.T_max)) / 2 
        return lr

    
if __name__ == '__main__':
    from vemol.optimizer.sgd import SGDConfig, SGDOptimizer
    cs = ConfigStore.instance()
    node = SGDConfig()
    name = 'sgd'
    node._name = name
    cs.store(name=name, group="optimizer", node=node)
    print(node)
    optimizer = SGDOptimizer(node)
    # model = torchvision.models.resnet18(pretrained=False)
    # print(optimizer.get_optimizer(model))
    # print(optimizer.get_scheduler(model))
