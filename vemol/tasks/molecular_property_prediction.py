import os
import sys
from dataclasses import dataclass, field
from typing import Any, List, Dict, Optional

from omegaconf import II, MISSING

from vemol.tasks import register_task
from vemol.dataclass.config import Config

from vemol.dataclass.config import CommonConfig, post_init
#from vemol.utils.monitor import monitor_config


defaults = [
    "_self_", # this is a special key that tells Hydra to use the parent key as the value
    {"dataset": "mol_graph_mpp"},
    {"optimizer": "adam"},
    {"checkpoint": "base_checkpoint"},
    {"model": "base_gnn"},
    {"criterion": "binary_cross_entropy"},
    {"metric": "binary_graph_cls"},
    {"scheduler": "fix"},
]


local_config = {
    'common':
    {
        'epochs': 50,
        'log_interval': 20, 
    },
}

# 似乎不能像yaml一样定义common，然后覆盖default common配置, 目前先用post_init函数来实现

@register_task('mpp')
@dataclass
class MPP(Config):
    defaults: List[Any] = field(default_factory=lambda: defaults)
    task: str = 'mpp' # molecular property prediction
    name: str = 'base'
    # common: Dict[Any, Any] = field(default_factory=lambda: {"epochs": 20})
    dataset: Any = MISSING 
    optimizer: Any = MISSING 
    checkpoint: Any = MISSING 
    model: Any = MISSING 
    criterion: Any = MISSING 
    metric: Any = MISSING 
    scheduler: Any = MISSING 
    local_config: Any = field(default_factory=lambda: local_config)
    #monitor: Any = field(default_factory=lambda: monitor_config)
    # def __post_init__(self):
    #     for k, v in local_config.items():
    #         assert k in self.__dataclass_fields__, f"{k} is not a valid field"
    #         sub_config = getattr(self, k)
    #         # print(k, v, sub_config)
    #         for sub_k, sub_v in v.items(): 
    #             assert sub_k in sub_config.__dataclass_fields__, f"{k}.{sub_k} is not a valid field"
    #             setattr(sub_config, sub_k, sub_v)
        # self.common.epochs = 20
        # self.common.validate = False
    
    # def __post_init__(self):
    #     post_init(self, local_config=local_config)