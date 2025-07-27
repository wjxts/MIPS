import os
import sys
from dataclasses import dataclass, field
from typing import Any, List, Dict, Optional

from omegaconf import II, MISSING

from vemol.tasks import register_task
from vemol.dataclass.config import Config

from vemol.dataclass.config import CommonConfig, post_init
from vemol.utils.monitor_config import MONITOR_MPP_GT_CONFIG


defaults = [
    "_self_", # this is a special key that tells Hydra to use the parent key as the value
    {"dataset": "mol_graph_mpp"},
    {"optimizer": "adam"},
    {"checkpoint": "base_checkpoint"},
    {"model": "base_graph_transformer"},
    {"criterion": "binary_cross_entropy"},
    {"metric": "binary_graph_cls"},
    {"scheduler": "fix"},
]


local_config = {
    'common':{
        'epochs': 50,
        'log_interval': 20, 
    },
    'dataset':{
        'completion': True, 
        'max_length': 3,
    },
    'model':{
        'num_layers': 2,
        'd_model': 128,
        'n_heads': 8
    }
}

# 似乎不能像yaml一样定义common，然后覆盖default common配置, 目前先用post_init函数来实现

@register_task('mpp_gt')
@dataclass
class MPP(Config):
    defaults: List[Any] = field(default_factory=lambda: defaults)
    task: str = 'mpp_gt' # molecular property prediction
    name: str = 'base'
    # common: Dict[Any, Any] = field(default_factory=lambda: {"epochs": 20})
    dataset: Any = MISSING 
    optimizer: Any = MISSING 
    checkpoint: Any = MISSING 
    model: Any = MISSING 
    criterion: Any = MISSING 
    metric: Any = MISSING 
    scheduler: Any = MISSING  
    monitor_config: Any = field(default_factory=lambda: MONITOR_MPP_GT_CONFIG)
    local_config: Any = field(default_factory=lambda: local_config)
    
   