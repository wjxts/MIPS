"""
存放自定义的module，供models通过cfg配置使用
存放基本的module，供models通过dataclass组装
"""
import os
import importlib

from hydra.core.config_store import ConfigStore

MODULE_REGISTRY = {}
MODULE_DATACLASS_REGISTRY = {}


def build_module(cfg, ext=None):
    module = MODULE_REGISTRY[cfg.module]
    if ext is None:
        return module(cfg)
    return module(cfg, ext)

def get_dataclass(module):
    return MODULE_DATACLASS_REGISTRY[module]


def register_module(name, dataclass=None):
    def register_module_cls(cls):
        if name in MODULE_REGISTRY:
            return MODULE_REGISTRY[name]

        MODULE_REGISTRY[name] = cls
        cls.__dataclass = dataclass
        if dataclass is not None:
            MODULE_DATACLASS_REGISTRY[name] = dataclass
            cs = ConfigStore.instance()
            node = dataclass()
            node._name = name

            cs.store(name=name, node=node, group='module')
        return cls

    return register_module_cls


def import_modules(modules_dir, namespace):
    for file in os.listdir(modules_dir):
        path = os.path.join(modules_dir, file)

        if (
                not file.startswith("_")
                and not file.startswith(".")
                and (file.endswith(".py") or os.path.isdir(path))
        ):
            module_name = file[: file.find(".py")] if file.endswith(".py") else file
            importlib.import_module(namespace + "." + module_name)


# automatically import any Python files in the models/ directory
# models_dir = os.path.dirname(__file__)
# import_modules(models_dir, "vemol.modules")

# for k, v in MODULE_REGISTRY.items():
#     print(k, v)
# print(len(MODULE_REGISTRY))
# print(get_module('Relu'))
