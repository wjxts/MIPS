import importlib
import os


def import_mol_graph_config(mol_graph_config_dir, namespace):
    for file in os.listdir(mol_graph_config_dir):
        path = os.path.join(mol_graph_config_dir, file)
        if (
            not file.startswith("_")
            and not file.startswith(".")
            and (file.endswith(".py") or os.path.isdir(path))
        ):
            mol_graph_config_name = file[: file.find(".py")] if file.endswith(".py") else file
            importlib.import_module(namespace + "." + mol_graph_config_name)


# automatically import any Python files in the models/ directory
mol_graph_config_dir = os.path.dirname(__file__)
import_mol_graph_config(mol_graph_config_dir, "vemol.dataset.mol_graph_config")