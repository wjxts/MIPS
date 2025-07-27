from dataclasses import is_dataclass

from vemol.dataclass import BaseDataclass
from omegaconf import DictConfig, OmegaConf, open_dict


def merge_with_parent(dc: BaseDataclass, cfg: DictConfig, remove_missing=False):
    if remove_missing:

        if is_dataclass(dc):
            target_keys = set(dc.__dataclass_fields__.keys())
        else:
            target_keys = set(dc.keys())

        with open_dict(cfg):
            for k in list(cfg.keys()):
                if k not in target_keys:
                    del cfg[k]

    merged_cfg = OmegaConf.merge(dc, cfg)
    merged_cfg.__dict__["_parent"] = cfg.__dict__["_parent"]
    OmegaConf.set_struct(merged_cfg, True)
    return merged_cfg
