import os
import sys
from dataclasses import dataclass, field
from typing import Any, List, Dict, Optional

import torch
from omegaconf import II, MISSING
from omegaconf import DictConfig



@dataclass
class BaseDataclass:
    """base dataclass that supported fetching attributes and metas"""

    _name: Optional[str] = None

    @staticmethod
    def name():
        return None

    def _get_all_attributes(self) -> List[str]:
        return [k for k in self.__dataclass_fields__.keys()]

    def _get_meta(
        self, attribute_name: str, meta: str, default: Optional[Any] = None
    ) -> Any:
        return self.__dataclass_fields__[attribute_name].metadata.get(meta, default)

    def _get_name(self, attribute_name: str) -> str:
        return self.__dataclass_fields__[attribute_name].name

    def _get_default(self, attribute_name: str) -> Any:
        if hasattr(self, attribute_name):
            if str(getattr(self, attribute_name)).startswith("${"):
                return str(getattr(self, attribute_name))
            elif str(self.__dataclass_fields__[attribute_name].default).startswith(
                "${"
            ):
                return str(self.__dataclass_fields__[attribute_name].default)
            elif (
                getattr(self, attribute_name)
                != self.__dataclass_fields__[attribute_name].default
            ):
                return getattr(self, attribute_name)

        f = self.__dataclass_fields__[attribute_name]
        if not isinstance(f.default_factory):
            return f.default_factory()
        return f.default

    def _get_type(self, attribute_name: str) -> Any:
        return self.__dataclass_fields__[attribute_name].type

    def _get_help(self, attribute_name: str) -> Any:
        return self._get_meta(attribute_name, "help")

    def _get_argparse_const(self, attribute_name: str) -> Any:
        return self._get_meta(attribute_name, "argparse_const")

    def _get_argparse_alias(self, attribute_name: str) -> Any:
        return self._get_meta(attribute_name, "argparse_alias")

    def _get_choices(self, attribute_name: str) -> Any:
        return self._get_meta(attribute_name, "choices")

    @classmethod
    def from_namespace(cls, args):
        if isinstance(args, cls):
            return args
        else:
            config = cls()
            for k in config.__dataclass_fields__.keys():
                if k.startswith("_"):
                    # private member, skip
                    continue
                if hasattr(args, k):
                    setattr(config, k, getattr(args, k))

            return config

@dataclass
class CommonConfig(BaseDataclass):
    num_threads: int = 8
    # train setting
    epochs: int = field(
        default=10,
        metadata={
            'help': "total epochs"
        }
    )
    max_steps: int = field(
        default=-1,
        metadata={
            'help': "max steps" # >0 表示激活max_steps设定
        }
    )
    validate: bool = field(
        default=True,
        metadata={
            'help': 'whether validate during training'
        }
    )
    start: int = field(
        default=1,
        metadata={
            'help': 'start epochs'
        }
    )
    # val setting 
    val_interval: int = 1
    # log setting
    log_interval: int = field(
        default=10,
        metadata={
            "help": "log progress every N batches (when progress bar is disabled)"
        },
    )
    log_dir: str = ''
    save_results: bool = False 
    save_results_dir: str = 'results'
    # visualize
    global_step: int = 1
    global_epoch: int = 1 # 应该没怎么用到, 可能在resume里涉及
    mixed_precision: bool = False   # dgl 1.x之后才支持mixed precision, 在conda dgl环境下可以运行, KPGT不行
    monitor: bool = False
    wandb: bool = False
    wandb_project: Optional[str] = field(
        default=None,
        metadata={"help": "Weights and Biases project name to use for logging"},
    )
    dump_monitor: bool = False
    # random seed
    seed: int = field(
        default=1, metadata={"help": "pseudo random number generator seed"}
    )
    # data_parallel
    data_parallel_size: int = field(
        default=1, metadata={"help": "total number of GPUs to parallelize data over"}
    )
    update_freq: int = 1
    ema_decay: float = field(
        default=-1, metadata={"help": "exponential moving average decay rate"}
    )
    device: int = field(
        default=0 if torch.cuda.is_available() else -1, # device=-1, x.to(device) will raise error
        metadata={"help": "device"}
    )



def post_init(obj: DictConfig, local_config: Dict):
    # 只支持一级递归赋值
    for k, v in local_config.items():
        assert k in obj, f"{k} is not a valid field"
        sub_config = getattr(obj, k)
        # print(k, v, sub_config)
        for sub_k, sub_v in v.items(): 
            # print(k, sub_k, sub_config)
            assert sub_k in sub_config, f"{k}.{sub_k} is not a valid field"
            setattr(sub_config, sub_k, sub_v)
            
@dataclass
class Config(BaseDataclass):
    # 默认配置
    common: CommonConfig = CommonConfig()
    dataset: Any = None
    optimizer: Any = None
    checkpoint: Any = None
    model: Any = None
    criterion: Any = None
    metric: Any = None
    monitor: Any = None
