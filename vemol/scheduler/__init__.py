import os
import importlib

from hydra.core.config_store import ConfigStore

from vemol.scheduler.__warmup import wrap_warmup
SCHEDULER_REGISTRY = {}
SCHEDULER_DATACLASS_REGISTRY = {}


def build_scheduler(cfg, optimizer=None):
    scheduler = SCHEDULER_REGISTRY[cfg.scheduler](cfg, optimizer)
    scheduler = wrap_warmup(scheduler, cfg)
    return scheduler


def register_scheduler(name, dataclass=None):
    def register_scheduler_cls(cls):
        if name in SCHEDULER_REGISTRY:
            return SCHEDULER_REGISTRY[name]

        SCHEDULER_REGISTRY[name] = cls
        cls.__dataclass = dataclass
        if dataclass is not None:
            SCHEDULER_DATACLASS_REGISTRY[name] = dataclass
            cs = ConfigStore.instance()
            node = dataclass()
            node._name = name
            cs.store(name=name, group="scheduler", node=node)
        return cls

    return register_scheduler_cls


def import_schedulers(schedulers_dir, namespace):
    for file in os.listdir(schedulers_dir):
        path = os.path.join(schedulers_dir, file)
        if (
                not file.startswith("_")
                and not file.startswith(".")
                and (file.endswith(".py") or os.path.isdir(path))
        ):
            scheduler_name = file[: file.find(".py")] if file.endswith(".py") else file
            importlib.import_module(namespace + "." + scheduler_name)


# automatically import any Python files in the models/ directory

schedulers_dir = os.path.dirname(__file__)
import_schedulers(schedulers_dir, "vemol.scheduler")
