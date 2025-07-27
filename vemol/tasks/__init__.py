import os
import importlib

from hydra.core.config_store import ConfigStore

TASK_REGISTRY = {}

def build_task(task_name):
    task = TASK_REGISTRY[task_name]
    return task

def register_task(name):
    def register_task_cls(cls):
        if name in TASK_REGISTRY:
            return TASK_REGISTRY[name]

        TASK_REGISTRY[name] = cls()
        cs = ConfigStore.instance()
        node = cls()
        node._name = name
        # print(name)
        cs.store(name=name, node=node)
        return cls

    return register_task_cls


def import_tasks(tasks_dir, namespace):
    for file in os.listdir(tasks_dir):
        path = os.path.join(tasks_dir, file)
        if (
                not file.startswith("_")
                and not file.startswith(".")
                and (file.endswith(".py") or os.path.isdir(path))
        ):
            task_name = file[: file.find(".py")] if file.endswith(".py") else file
            importlib.import_module(namespace + "." + task_name)


# automatically import any Python files in the models/ directory
tasks_dir = os.path.dirname(__file__)
import_tasks(tasks_dir, "vemol.tasks")