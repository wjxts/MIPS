import importlib
import os

def import_smiles2graph(smiles2graph_dir, namespace):
    for file in os.listdir(smiles2graph_dir):
        path = os.path.join(smiles2graph_dir, file)
        if (
            not file.startswith("_")
            and not file.startswith(".")
            and (file.endswith(".py") or os.path.isdir(path))
        ):
            smiles2graph_name = file[: file.find(".py")] if file.endswith(".py") else file
            importlib.import_module(namespace + "." + smiles2graph_name)

# automatically import any Python files in the models/ directory
# datasets_dir = os.path.dirname(__file__)
# import_smiles2graph(datasets_dir, "vemol.chem_utils.smiles2graph")

# Python 只会加载你显式导入的模块及其父级模块


