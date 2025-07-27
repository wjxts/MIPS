import os
import importlib

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import all_reduce

from hydra.core.config_store import ConfigStore
import omegaconf

MODEL_REGISTRY = {}
MODEL_DATACLASS_REGISTRY = {}


def build_model(cfg):
    Model = MODEL_REGISTRY[cfg.model]
    model = Model(cfg)
    model.move_model_to_gpu()
    return model


def register_model(name, dataclass=None):
    def register_model_cls(cls):
        if name in MODEL_REGISTRY:
            return MODEL_REGISTRY[name]

        MODEL_REGISTRY[name] = cls
        cls.__dataclass = dataclass
        if dataclass is not None:
            MODEL_DATACLASS_REGISTRY[name] = dataclass
            cs = ConfigStore.instance()
            node = dataclass()
            node._name = name
            cs.store(name=name, group="model", node=node)
        return cls

    return register_model_cls


def import_models(models_dir, namespace):
    for file in os.listdir(models_dir):
        path = os.path.join(models_dir, file)
        if (
                not file.startswith("_")
                and not file.startswith(".")
                and (file.endswith(".py") or os.path.isdir(path))
        ):
            model_name = file[: file.find(".py")] if file.endswith(".py") else file
            importlib.import_module(namespace + "." + model_name) # import 之后就会去读文件，调用里边的register


# automatically import any Python files in the models/ directory
models_dir = os.path.dirname(__file__)
import_models(models_dir, "vemol.models")




