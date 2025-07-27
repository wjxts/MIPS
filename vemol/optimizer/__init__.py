import os
import importlib

from hydra.core.config_store import ConfigStore

OPTIMIZER_REGISTRY = {}
OPTIMIZER_DATACLASS_REGISTRY = {}


def build_optimizer(cfg):
    optimizer = OPTIMIZER_REGISTRY[cfg.optimizer](cfg)
    return optimizer


def register_optimizer(name, dataclass=None):
    def register_optimizer_cls(cls):
        if name in OPTIMIZER_REGISTRY:
            return OPTIMIZER_REGISTRY[name]

        OPTIMIZER_REGISTRY[name] = cls
        cls.__dataclass = dataclass
        if dataclass is not None:
            OPTIMIZER_DATACLASS_REGISTRY[name] = dataclass
            cs = ConfigStore.instance()
            node = dataclass()
            node._name = name
            cs.store(name=name, group="optimizer", node=node)
        return cls

    return register_optimizer_cls


def import_optimizers(optimizers_dir, namespace):
    for file in os.listdir(optimizers_dir):
        path = os.path.join(optimizers_dir, file)
        if (
                not file.startswith("_")
                and not file.startswith(".")
                and (file.endswith(".py") or os.path.isdir(path))
        ):
            optimizer_name = file[: file.find(".py")] if file.endswith(".py") else file
            importlib.import_module(namespace + "." + optimizer_name)


# automatically import any Python files in the models/ directory
optimizers_dir = os.path.dirname(__file__)
import_optimizers(optimizers_dir, "vemol.optimizer")
