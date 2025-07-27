from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore

MOL_GRAPH_FORM_DATACLASS_REGISTRY = {}

# name list:
# atom_graph, fragment_graph

def register_mol_graph_form_config(name):
    def register_mol_graph_form_cls(dataclass):    
        MOL_GRAPH_FORM_DATACLASS_REGISTRY[name] = dataclass
        cs = ConfigStore.instance()
        node = dataclass()
        node._name = name
        cs.store(name=name, group="dataset/mol_graph_form_cfg", node=node)
        return dataclass
    return register_mol_graph_form_cls


# @register_mol_graph_form_config('base')
# @dataclass
# class BaseGraphConfig:
#     name: str = field(
#         default='base', metadata={'help': "The name of graph_form_cfg."}
#     )

@register_mol_graph_form_config('dummy_graph')
@dataclass
class BaseGraphConfig:
    # 保存都要实现的变量
    name: str = field(
        default='dummy_graph', metadata={'help': "The name of graph_form_cfg."}
    )
    custom_name: str = field(
        default='dummy_graph', metadata={'help': "The custom name of graph_form_cfg."}
    )

@register_mol_graph_form_config('atom_graph')
@dataclass
class AtomGraphConfig(BaseGraphConfig):
    name: str = field(
        default='atom_graph', metadata={'help': "The name of graph_form_cfg."}
    )
    custom_name: str = field(
        default='atom_graph', metadata={'help': "The custom name of graph_form_cfg."}
    )
    

@register_mol_graph_form_config('fragment_graph')
@dataclass
class FragmentGraphConfig(AtomGraphConfig):
    name: str = field(
        default='fragment_graph', metadata={'help': "The name of graph_form_cfg."}
    )
    custom_name: str = field(
        default='dove_graph', metadata={'help': "The custom name of graph_form_cfg."}
    )
    vocab_path: str = field(
        default='', metadata={'help': "vocab path."}
    )
    order: int = field(
        default=0, metadata={'help': "order of line graph."}
    )
    atom_graph_lmdb_path: str = field(
        default='', metadata={'help': "The cache path of atom graph dataset."}
    )
    fragment_graph_lmdb_path: str = field(
        default='', metadata={'help': "The cache path of fragment graph dataset. used in tokenizer."}
    )
  

@register_mol_graph_form_config('spatial_graph')
@dataclass
class SpatialGraphConfig(BaseGraphConfig):
    name: str = field(
        default='spatial_graph', metadata={'help': "The name of graph_form_cfg."}
    )
    custom_name: str = field(
        default='spatial_graph', metadata={'help': "The custom name of graph_form_cfg."}
    )
    property: str = 'all'
    explicit_H: bool = True


@register_mol_graph_form_config('polymer_atom_graph')
@dataclass
class PolymerAtomGraphConfig(AtomGraphConfig):
    name: str = field(
        default='polymer_atom_graph', metadata={'help': "The name of graph_form_cfg."}
    )
    custom_name: str = field(
        default='polymer_atom_graph', metadata={'help': "The custom name of graph_form_cfg."}
    )
    op: str = 'star_keep'
    main_chain_embed: bool = field(
        default=False, metadata={'help': "whether to add main chain embedding."}
    )

@register_mol_graph_form_config('polymer_fragment_graph')
@dataclass
class PolymerFragmentGraphConfig(PolymerAtomGraphConfig, FragmentGraphConfig):
    name: str = field(
        default='polymer_fragment_graph', metadata={'help': "The name of graph_form_cfg."}
    )
    custom_name: str = field(
        default='polymer_fragment_graph', metadata={'help': "The custom name of graph_form_cfg."}
    )
    order: int = field(
        default=1, metadata={'help': "order of line graph."}
    )
    # 仅在下游任务使用, 不需要cache
    

