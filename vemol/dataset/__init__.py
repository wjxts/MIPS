import importlib
import os

from hydra.core.config_store import ConfigStore


DATASET_REGISTRY = {}
DATASET_DATACLASS_REGISTRY = {}


def build_dataloader(cfg):
    dataset = DATASET_REGISTRY[cfg.dataset](cfg)
    return dataset


def register_dataset(name, dataclass=None):
    def register_dataset_cls(cls):
        if name in DATASET_REGISTRY:
            return DATASET_REGISTRY[name]

        DATASET_REGISTRY[name] = cls
        cls.__dataclass = dataclass
        if dataclass is not None:
            DATASET_DATACLASS_REGISTRY[name] = dataclass
            cs = ConfigStore.instance()
            node = dataclass()
            node._name = name
            cs.store(name=name, group="dataset", node=node)
        return cls

    return register_dataset_cls


def import_datasets(datasets_dir, namespace):
    for file in os.listdir(datasets_dir):
        path = os.path.join(datasets_dir, file)
        if (
            not file.startswith("_")
            and not file.startswith(".")
            and (file.endswith(".py") or os.path.isdir(path))
        ):
            dataset_name = file[: file.find(".py")] if file.endswith(".py") else file
            importlib.import_module(namespace + "." + dataset_name)


# automatically import any Python files in the models/ directory
datasets_dir = os.path.dirname(__file__)
import_datasets(datasets_dir, "vemol.dataset")