
from vemol.scheduler.base_scheduler import BaseScheduler, BaseSchedulerConfig

# 定义完dataloader才知道warmup_steps的具体值

def wrap_warmup(scheduler: BaseScheduler, cfg: BaseSchedulerConfig):
    warmup_steps = cfg.warmup_steps
    warmup_epochs = cfg.warmup_epochs
    num_batches_per_epoch = cfg.num_batches_per_epoch
    if (warmup_epochs<0 and warmup_steps<0) or (not scheduler.cfg.support_warmup_wrapper):
        print(f"no learning rate warmup needed")
        return scheduler
    else:
        print(f"wrap learning rate warmup")
        if warmup_epochs>0:
            warmup_steps = warmup_epochs * num_batches_per_epoch
        return WarmUpWrapper(scheduler, warmup_steps, warmup_epochs)

class WarmUpWrapper(BaseScheduler):
    def __init__(self, scheduler: BaseScheduler, warmup_steps: int=-1, warmup_epochs: int=-1):
        self.warmup_steps = warmup_steps
        self.warmup_epochs = warmup_epochs
        self.max_lr = scheduler.cfg.lr
        super().__init__(scheduler.cfg, scheduler.optimizer)
        self.scheduler = scheduler
    
    def get_step_lr(self, step: int):
        if step <= self.warmup_steps:
            self.lr = self.max_lr  * step / self.warmup_steps
        else:
            self.lr = self.scheduler.get_step_lr(step)
        return self.lr
    
    def get_epoch_lr(self, epoch: int):
        if epoch <= self.warmup_epochs:
            self.lr = self.max_lr * epoch / self.warmup_epochs
        else:
            self.lr = self.scheduler.get_epoch_lr(epoch)
        return self.lr
    
    def state_dict(self):
        d = self.scheduler.state_dict()
        d['warmup_steps'] = self.warmup_steps
        d['warmup_epochs'] = self.warmup_epochs
        return d 
    
    def load_state_dict(self, state_dict):
        self.warmup_steps = state_dict['warmup_steps']
        self.warmup_epochs = state_dict['warmup_epochs']
        self.scheduler.load_state_dict(state_dict)
    