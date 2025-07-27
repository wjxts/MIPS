import os
import sys
from dataclasses import dataclass, field
from typing import Any, List, Optional

from omegaconf import II, MISSING

from vemol.tasks import register_task
from vemol.dataclass.config import Config, post_init

#from vemol.utils.monitor import monitor_config


defaults = [
    
    {"dataset": "mol_graph_ssl"},
    {"optimizer": "adam"},
    {"checkpoint": "base_checkpoint"},
    {"model": "base_gnn"},
    {"criterion": "cross_entropy_mae"},
    {"metric": "dummy"},
    {"scheduler": "poly_decay"},
    "_self_", # this is a special key that tells Hydra to use the parent key as the value
]

local_config = {
    
    'common':{
            'validate': False,
            'log_interval': 50, 
            'epochs': 1,
        },
    'model': {
            'readout': 'none',
        },
    'dataset':{
        'num_workers': 8,
    },
    'checkpoint':{
        'save_step': True,
        'save_epoch': False, 
    }
}


@register_task('mae_pretrain')
@dataclass
class MAEPretrain(Config):
    defaults: List[Any] = field(default_factory=lambda: defaults)
    task: str = 'mae_pretrain' # molecular mae pre-training
    name: str = 'base' # 是必须的字段, 会在checkpoint中用
    dataset: Any = MISSING 
    optimizer: Any = MISSING 
    checkpoint: Any = MISSING 
    model: Any = MISSING 
    criterion: Any = MISSING 
    metric: Any = MISSING 
    scheduler: Any = MISSING 
    # monitor: Any = field(default_factory=lambda: monitor_config)
    local_config: Any = field(default_factory=lambda: local_config)
    # def __post_init__(self):
    #     post_init(self, local_config=local_config)