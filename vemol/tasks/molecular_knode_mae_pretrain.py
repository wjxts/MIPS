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
    {"model": "knode_graph_transformer"},
    {"criterion": "cross_entropy_kvec_mae"},
    {"metric": "dummy"},
    # {"scheduler": "fix"},
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
    'checkpoint':{
        'save_step': True,
        'save_epoch': False, 
    },
    'dataset':{
        'max_length': 3,
        'kvec_names': ['md', 'rdkfp'],
        'add_knodes': True, 
        'completion': True,
        'num_workers': 8,
    },
    'criterion':{
        'weight_path': '/mnt/nvme_share/wangjx/mol_repre/KPGT/datasets/chembl29/descriptor'
    },
}


@register_task('knode_mae_pretrain')
@dataclass
class MAEPretrain(Config):
    defaults: List[Any] = field(default_factory=lambda: defaults)
    task: str = 'knode_mae_pretrain' # molecular mae pre-training
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