import os
import logging 
from torch.optim.lr_scheduler import *
from builtins import str
from dataclasses import field, dataclass
from omegaconf import II, MISSING
from typing import Any, List, Optional
import numpy as np

# import warmup_scheduler
from abc import ABC, abstractmethod

from vemol.dataclass import BaseDataclass

logger = logging.getLogger(__name__)

@dataclass
class OptimizerConfig(BaseDataclass):
    lr: List[float] = field(
        default_factory=lambda: [0.001],
        metadata={
            "help": "learning rate for the first N epochs; all epochs >N using LR_N"
            " (note: this may be interpreted differently depending on --lr-scheduler)"
        },
    )
    epochs: int = II("common.epochs")
    optimizer: str = field(
        default="must be specify", metadata={"help": "the optimizer method to train network"}
    )
    weight_decay: float = field(
        default=0, metadata={"help": "weight decay (default: 0)."}
    )
    clip_norm: float = -1.0


class Optimizer(ABC):
    def __init__(self, cfg: OptimizerConfig):
        self.cfg = cfg
        self.optimizer = None

    def _get_model_grouped_parameters(self, model):
        # return model.parameters()
        model_parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
        n_params = sum([np.prod(p.size()) for p in model_parameters])
        logger.info("Trainable parameters: {}".format(n_params))
        return model_parameters

    @abstractmethod
    def _build_optimizer(self, model):
        """
        子类必须实现的方法
        """
        pass
        # params = self._get_model_grouped_parameters(model)
        # return self.optimizer

    def get_optimizer(self):
        return self.optimizer

    def set_model(self, model):
        self._build_optimizer(model)

    def step(self):
        return self.optimizer.step()

    def zero_grad(self):
        return self.optimizer.zero_grad()

    def get_param_lr(self, groups_id=0):
        return self.optimizer.param_groups[groups_id]["lr"]
