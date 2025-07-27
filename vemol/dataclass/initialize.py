import logging
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from vemol.tasks import build_task

logger = logging.getLogger(__name__)


def hydra_init(task_name="mpp") -> None:
    task = build_task(task_name)
    cs = ConfigStore.instance()
    cs.store(name=f"{task_name}", node=task)
    # print(task)

    for k in task.__dataclass_fields__:
        v = task.__dataclass_fields__[k].default
        try:
            cs.store(name=k, node=v)
        except BaseException:
            logger.error(f"{k} - {v}")
            raise

Config = {} # unknown config, not used!

def add_defaults(cfg: DictConfig) -> None:
    """This function adds default values that are stored in dataclasses that hydra doesn't know about"""

    from vemol.tasks import TASK_DATACLASS_REGISTRY
    from vemol.models import MODEL_DATACLASS_REGISTRY
    from vemol.dataclass.utils import merge_with_parent
    from typing import Any

    OmegaConf.set_struct(cfg, False)

    for k, v in Config.__dataclass_fields__.items():
        field_cfg = cfg.get(k)
        if field_cfg is not None and v.type == Any:
            dc = None

            if isinstance(field_cfg, str):
                field_cfg = DictConfig({"_name": field_cfg})
                field_cfg.__dict__["_parent"] = field_cfg.__dict__["_parent"]

            name = getattr(field_cfg, "_name", None)

            if k == "task":
                dc = TASK_DATACLASS_REGISTRY.get(name)
            elif k == "model":
                dc = MODEL_DATACLASS_REGISTRY.get(name)

            if dc is not None:
                cfg[k] = merge_with_parent(dc, field_cfg)
    
    OmegaConf.set_struct(cfg, True)
