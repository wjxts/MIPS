
from dataclasses import dataclass, field
from omegaconf import II, MISSING

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from vemol.scheduler import register_scheduler
from vemol.scheduler.base_scheduler import BaseSchedulerConfig, BaseScheduler
from vemol.scheduler.base_scheduler import PER_STEP_MODE

class PolynomialDecayLR(_LRScheduler):

    def __init__(self, optimizer, warmup_updates, tot_updates, lr, end_lr, power, last_epoch=-1, verbose=False):
        self.warmup_updates = warmup_updates
        self.tot_updates = tot_updates
        self.lr = lr
        self.end_lr = end_lr
        self.power = power
        super(PolynomialDecayLR, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if self._step_count <= self.warmup_updates:
            self.warmup_factor = self._step_count / self.warmup_updates
            lr = self.warmup_factor * self.lr
        elif self._step_count >= self.tot_updates:
            lr = self.end_lr
        else:
            warmup = self.warmup_updates
            lr_range = self.lr - self.end_lr
            pct_remaining = 1 - (self._step_count - warmup) / (
                self.tot_updates - warmup
            )
            lr = lr_range * pct_remaining ** (self.power) + self.end_lr
        return [lr for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        assert False
        
@dataclass
class PolynomialDecayConfig(BaseSchedulerConfig):
    scheduler: str = field(
        default='poly_decay', metadata={"help": "the scheduler method to modify lr)"}
    )
    lr: float = field(
        default=1e-3, metadata={"help": "max learning rate"}
    )
    end_lr: float = field(
        default=1e-9, metadata={"help": "end learning rate"}
    )
    warmup_steps: int = field(
        default=1000, metadata={"help": "warmup steps"}
    )
    total_steps: int = II("common.max_steps")
    power: int = 1


@register_scheduler('poly_decay', PolynomialDecayConfig)
class PolynomialDecayScheduler(BaseScheduler):
    def __init__(self, cfg: PolynomialDecayConfig, optimizer: Optimizer):
        assert cfg.update_type == PER_STEP_MODE, f"update_type={cfg.update_type} is not supported for polynomial decay scheduler!"
        assert cfg.warmup_steps > 0 or cfg.warmup_epochs > 0, f"cfg.warmup_steps={cfg.warmup_steps} or cfg.warmup_epochs={cfg.warmup_epochs} should be greater than 0!"
        self.end_lr = cfg.end_lr
        self.power = cfg.power
        self.max_lr = cfg.lr
        self.total_steps = cfg.total_steps
        super().__init__(cfg, optimizer)
        
    # def _build_scheduler(self): # 可以不写
    #     return self
        # return PolynomialDecayLR(self.optimizer, 
        #                          warmup_updates=self.cfg.warmup_steps, 
        #                          tot_updates=self.cfg.total_steps, 
        #                          lr=self.cfg.lr, 
        #                          end_lr=self.cfg.end_lr,
        #                          power=self.cfg.power)
    
    def get_step_lr(self, step: int) -> float:
        if step >= self.total_steps:
            lr = self.end_lr
        else:
            lr_range = self.max_lr - self.end_lr
            pct_remaining = 1 - (step-self.cfg.warmup_steps)/(self.total_steps-self.cfg.warmup_steps)
            lr = lr_range * pct_remaining**self.power + self.end_lr
        return lr 
    
    def get_epoch_lr(self, epoch: int) -> float:
        return None 

