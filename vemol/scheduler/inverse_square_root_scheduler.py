from vemol.dataclass import BaseDataclass
from vemol.scheduler.base_scheduler import BaseSchedulerConfig, BaseScheduler
from dataclasses import dataclass, field
from omegaconf import II, MISSING
from vemol.scheduler import register_scheduler
from torch.optim import Optimizer

@dataclass
class InverseSquareRootSchedulerConfig(BaseSchedulerConfig):
    scheduler: str = field(
        default='inverse_square_root', metadata={"help": "the scheduler method to modify lr)"}
    )
    lr: float = field(
        default=5e-4, metadata={"help": "max learning rate"}
    )
    # warmup_init_lr: float = field(
    #     default=0., metadata={"help": "warmup initial learning rate"}
    # )
    # warmup_steps: int = field(
    #     default=4000, metadata={"help": "warmup steps"}
    # )


@register_scheduler('inverse_square_root', InverseSquareRootSchedulerConfig)
class InverseSquareRootScheduler(BaseScheduler):
    def __init__(self, cfg: InverseSquareRootSchedulerConfig, optimizer: Optimizer):
        super().__init__(cfg, optimizer)
  
        # self.warmup_end_lr = cfg.lr
        # self.lr_step = (self.warmup_end_lr - cfg.warmup_init_lr) / cfg.warmup_updates
        # self.decay_factor = self.warmup_end_lr * cfg.warmup_updates ** 0.5
        # self.lr = cfg.warmup_init_lr
        
    
    def get_step_lr(self, step: int) -> float:
        return self.lr
    
    def get_epoch_lr(self, epoch: int) -> float:
        return None
    
    def state_dict(self):
        return {"cfg": self.cfg}
    
    def load_state_dict(self, state_dict):
        self.cfg = state_dict["cfg"]
        #self.__init__(self.cfg, optimizer=None)
        

