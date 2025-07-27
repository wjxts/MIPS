import importlib
import os

from hydra.core.config_store import ConfigStore


MOL_GRAPH_COLLATOR_REGISTRY = {}


# def build_mol_graph_collator(cfg):
#     mol_collator = MOL_GRAPH_COLLATOR_REGISTRY[cfg.name](cfg)
    # return mol_collator

def get_mol_graph_collator_cls(cfg):
    collator_cfg = cfg.mol_graph_collator_cfg
    graph_form_cfg = cfg.mol_graph_form_cfg
    collator_name = collator_cfg.name
    graph_form = graph_form_cfg.name
    # mol_collator_cls = MOL_GRAPH_COLLATOR_REGISTRY[f"{graph_form}_{collator_name}"] # e.g., {atom_graph}_{mpp}

    if graph_form == 'polymer_atom_graph' and collator_name == 'mpp':
        mol_collator_cls = MOL_GRAPH_COLLATOR_REGISTRY['polymer_atom_graph_mpp']
    elif graph_form == 'polymer_atom_graph' and collator_name == 'mae':
        mol_collator_cls = MOL_GRAPH_COLLATOR_REGISTRY['polymer_atom_graph_mae']
    elif graph_form == 'polymer_fragment_graph' and collator_name == 'mpp':
        mol_collator_cls = MOL_GRAPH_COLLATOR_REGISTRY['polymer_fragment_graph_mpp']
    else:
        raise ValueError(f"Unsupported graph_form: {graph_form} and collator_name: {collator_name}")
        
    return mol_collator_cls

def register_mol_graph_collator(name):
    def register_mol_graph_collator_cls(cls):
        if name in MOL_GRAPH_COLLATOR_REGISTRY:
            return MOL_GRAPH_COLLATOR_REGISTRY[name]

        MOL_GRAPH_COLLATOR_REGISTRY[name] = cls
        return cls
    return register_mol_graph_collator_cls


def import_mol_graph_collators(mol_graph_collators_dir, namespace):
    for file in os.listdir(mol_graph_collators_dir):
        path = os.path.join(mol_graph_collators_dir, file)
        if (
            not file.startswith("_")
            and not file.startswith(".")
            and (file.endswith(".py") or os.path.isdir(path))
        ):
            mol_collator_name = file[: file.find(".py")] if file.endswith(".py") else file
            importlib.import_module(namespace + "." + mol_collator_name)


# automatically import any Python files in the models/ directory
mol_graph_collators_dir = os.path.dirname(__file__)
import_mol_graph_collators(mol_graph_collators_dir, "vemol.dataset.mol_graph_config.mol_graph_collator")