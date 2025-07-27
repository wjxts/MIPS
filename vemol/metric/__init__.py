import os
import importlib

from hydra.core.config_store import ConfigStore

METRIC_REGISTRY = {}
METRIC_DATACLASS_REGISTRY = {}


def build_metric(cfg):
    metric = METRIC_REGISTRY[cfg.metric](cfg)
    return metric


def register_metric(name, dataclass=None):
    def register_metric_cls(cls):
        if name in METRIC_REGISTRY:
            return METRIC_REGISTRY[name]

        METRIC_REGISTRY[name] = cls
        cls.__dataclass = dataclass
        if dataclass is not None:
            METRIC_DATACLASS_REGISTRY[name] = dataclass
            cs = ConfigStore.instance()
            node = dataclass()
            node._name = name
            cs.store(name=name, group="metric", node=node)
        return cls

    return register_metric_cls


def import_metrics(metrics_dir, namespace):
    for file in os.listdir(metrics_dir):
        path = os.path.join(metrics_dir, file)
        if (
                not file.startswith("_")
                and not file.startswith(".")
                and (file.endswith(".py") or os.path.isdir(path))
        ):
            metric__name = file[: file.find(".py")] if file.endswith(".py") else file
            importlib.import_module(namespace + "." + metric__name)


# automatically import any Python files in the models/ directory
metric_dir = os.path.dirname(__file__)
import_metrics(metric_dir, "vemol.metric")
