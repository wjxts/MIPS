from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore


MOL_COLLATOR_DATACLASS_REGISTRY = {}

# name list:
# mpp, mae, cl

def register_mol_graph_collator_config(name):
    def register_mol_collator_cls(dataclass):    
        MOL_COLLATOR_DATACLASS_REGISTRY[name] = dataclass
        cs = ConfigStore.instance()
        node = dataclass()
        node._name = name
        cs.store(name=name, group="dataset/mol_graph_collator_cfg", node=node)
        return dataclass
    return register_mol_collator_cls


@register_mol_graph_collator_config('mpp')
@dataclass
class BaseCollatorConfig:
    name: str = field(
        default='mpp', metadata={'help': "The name of mol_graph_collator_cfg."}
    )
    lmdb_path: str = field(
        default='', metadata={'help': "The cache path of dataset."}
    )

@register_mol_graph_collator_config('mae')
@dataclass
class MAECollatorConfig:
    name: str = field(
        default='mae', metadata={'help': "The name of mol_graph_collator_cfg."}
    )
    downsize: int = field(
        default=5000, metadata={'help': "The size of dataset. -1 means all. "}
    )
    lmdb_path: str = field(
        default='', metadata={'help': "The cache path of dataset."}
    )
    kvec_mask_rate: float = field(
        default=0.3, metadata={'help': "The rate of kvec masking."}
    )
    atom_mask_rate: float = field(
        default=0.3, metadata={'help': "The rate of atom masking."}
    )
    fragment_node_mask_rate: float = field(
        default=0.3, metadata={'help': "The rate of macro node masking."}
    )

@register_mol_graph_collator_config('cl')
@dataclass
class CLCollatorConfig(MAECollatorConfig):
    name: str = field(
        default='cl', metadata={'help': "The name of mol_graph_collator_cfg."}
    )
    atom_mask_rate: float = field(
        default=0.25, metadata={'help': "The rate of atom masking."}
    )
    bond_mask_rate: float = field(
        default=0.25, metadata={'help': "The rate of bond masking."}
    )