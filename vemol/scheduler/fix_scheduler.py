import numpy as np
from dataclasses import dataclass, field

from torch.optim.lr_scheduler import CosineAnnealingLR
from omegaconf import II, MISSING
from hydra.core.config_store import ConfigStore

from vemol.scheduler import register_scheduler
from vemol.scheduler.base_scheduler import BaseSchedulerConfig, BaseScheduler
from vemol.dataclass import BaseDataclass

@dataclass
class FixSchedulerConfig(BaseSchedulerConfig):
    scheduler: str = field(
        default='fix', metadata={"help": "the scheduler method to modify lr)"}
    )

@register_scheduler('fix', FixSchedulerConfig)
class FixScheduler(BaseScheduler):
    def __init__(self, cfg: FixSchedulerConfig, optimizer=None):
        super().__init__(cfg, optimizer)
        
    def get_step_lr(self, step: int) -> float:
        return self.cfg.lr
    
    def get_epoch_lr(self, epoch: int) -> float:
        return self.cfg.lr
        
if __name__ == '__main__':
    pass
