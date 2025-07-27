import os
import importlib

from hydra.core.config_store import ConfigStore

CRITERION_REGISTRY = {}
CRITERION_DATACLASS_REGISTRY = {}


def build_criterion(cfg):
    criterion = CRITERION_REGISTRY[cfg.criterion](cfg)
    return criterion


def register_criterion(name, dataclass=None):
    def register_criterion_cls(cls):
        if name in CRITERION_REGISTRY:
            return CRITERION_REGISTRY[name]

        CRITERION_REGISTRY[name] = cls
        cls.__dataclass = dataclass
        if dataclass is not None:
            CRITERION_DATACLASS_REGISTRY[name] = dataclass
            cs = ConfigStore.instance()
            node = dataclass()
            node._name = name
            cs.store(name=name, group="criterion", node=node)
        return cls

    return register_criterion_cls


def import_criterions(criterions_dir, namespace):
    for file in os.listdir(criterions_dir):
        path = os.path.join(criterions_dir, file)
        if (
                not file.startswith("_")
                and not file.startswith(".")
                and (file.endswith(".py") or os.path.isdir(path))
        ):
            criterion_name = file[: file.find(".py")] if file.endswith(".py") else file
            importlib.import_module(namespace + "." + criterion_name)


# automatically import any Python files in the models/ directory
criterions_dir = os.path.dirname(__file__)
import_criterions(criterions_dir, "vemol.criterion")
