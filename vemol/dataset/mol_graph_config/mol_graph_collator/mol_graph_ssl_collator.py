

from omegaconf import DictConfig

import dgl
import numpy as np
import torch 

from vemol.dataset.mol_graph_config.mol_graph_collator import register_mol_graph_collator
from vemol.chem_utils.featurizer.standard_featurizer import ATOM_NUMS
from vemol.dataset.mol_graph_config.mol_graph_collator.mol_graph_mpp_collator import batch_fragment_graphs, batch_atom_graphs

from vemol.chem_utils.fingerprint.fingerprint import FP_DIM, disturb_fp

from vemol.chem_utils.augmentor.base_mol_augmentor import BaseMolAugmentor

# @register_mol_graph_collator('atom_graph_mae')
class __AtomGraphMAECollator(object):
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.atom_mask_rate = cfg.atom_mask_rate

    def generate_mask(self, bg, mask_rate): # mask=1表示要mask掉
        
        mask = torch.bernoulli(torch.full((bg.num_nodes(),), mask_rate, device=bg.device))
        # mask = mask.bool()
        mask = mask.float()
        return mask 
        
    def __call__(self, item_list):
        data = batch_atom_graphs(item_list)
        batch_graph = data['atom_graph']
        labels = torch.argmax(batch_graph.ndata['h'][:, :ATOM_NUMS], dim=1)
        atom_mask = self.generate_mask(batch_graph, self.atom_mask_rate)
        batch_graph.ndata['h'] = batch_graph.ndata['h']*(1-atom_mask[:, None])
        data['labels'] = {'labels':labels, 'atom_mask':atom_mask} # mask要用来算loss
        return data

# @register_mol_graph_collator('fragment_graph_mae')
class __FragmentGraphMAECollator(__AtomGraphMAECollator):
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.fragment_node_mask_rate = cfg.mol_graph_collator_cfg.fragment_node_mask_rate
        
    def __call__(self, item_list):
        data = batch_fragment_graphs(item_list)
        labels = data['fragment_graph'].ndata['detail_idxs']
        fragment_node_mask = self.generate_mask(data['fragment_graph'], self.fragment_node_mask_rate)
        data['labels'] = {'labels':labels, 
                          'fragment_node_mask':fragment_node_mask, 
                          'mask':fragment_node_mask} # mask要用来算loss
        return data

@register_mol_graph_collator('atom_graph_mae') 
class AtomGraphMAECollator(object):
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.atom_mask_rate = cfg.mol_graph_collator_cfg.atom_mask_rate
        self.kvec_mask_rate = cfg.mol_graph_collator_cfg.kvec_mask_rate

    def generate_mask(self, bg, mask_rate):
        
        mask = torch.bernoulli(torch.full((bg.num_nodes(),), mask_rate, device=bg.device))
        # mask = mask.bool()
        mask = mask.float()
        # 不要mask knode
        if self.cfg.add_knodes:
            node_indicator = bg.ndata['node_indicator']
            mask[node_indicator>=1] = 0
        return mask 
    
    def get_batch_kvecs(self, kvecs_list):
        batch_kvecs = {}
        batch_disturb_kvecs = {}
        batch_dicturb_kvecs_mask = {}
        for name in kvecs_list[0]:
            batch_kvec = np.array([kvecs[name] for kvecs in kvecs_list]).astype(np.float32)
            batch_disturb_kvec, kvec_mask = disturb_fp(batch_kvec, name, self.kvec_mask_rate)
            batch_kvecs[name] = torch.from_numpy(batch_kvec) 
            batch_disturb_kvecs[name] = torch.from_numpy(batch_disturb_kvec)
            batch_dicturb_kvecs_mask[name] = torch.from_numpy(kvec_mask)
        return batch_kvecs, batch_disturb_kvecs, batch_dicturb_kvecs_mask
    
    def __call__(self, item_list):
        graphs, kvecs_list = list(zip(*item_list))
        batch_graph = dgl.batch(graphs)
        labels = torch.argmax(batch_graph.ndata['h'][:, :ATOM_NUMS], dim=1)
        atom_mask = self.generate_mask(batch_graph, self.atom_mask_rate)
        
        batch_kvecs, batch_disturb_kvecs, batch_dicturb_kvecs_mask = self.get_batch_kvecs(kvecs_list)
        
        batch_graph.ndata['h'] = batch_graph.ndata['h']*(1-atom_mask[:, None])
        data = {}
        data['atom_graph'] = batch_graph
        data['kvecs'] = batch_disturb_kvecs # label与kvec_graph_mpp统一
        data['labels'] = {'labels':labels, 
                          'atom_mask':atom_mask,
                          'kvecs_labels': batch_kvecs,
                          'kvecs_mask': batch_dicturb_kvecs_mask} # mask要用来算loss
        return data
    
    
@register_mol_graph_collator('fragment_graph_mae')
class FragmentGraphMAECollator(AtomGraphMAECollator):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)
        self.fragment_node_mask_rate = cfg.mol_graph_collator_cfg.fragment_node_mask_rate

    def __call__(self, item_list):
        graph_items, kvecs_list = list(zip(*item_list))
        
        data = batch_fragment_graphs(graph_items)
        labels = data['fragment_graph'].ndata['detail_idxs']
        fragment_node_mask = self.generate_mask(data['fragment_graph'], self.fragment_node_mask_rate)
        
        
        batch_kvecs, batch_disturb_kvecs, batch_dicturb_kvecs_mask = self.get_batch_kvecs(kvecs_list)
    
        data['kvecs'] = batch_disturb_kvecs
        
        data['labels'] = {'labels':labels, 
                          'fragment_node_mask':fragment_node_mask, 
                          'mask':fragment_node_mask,
                          'kvecs_labels': batch_kvecs,
                          'kvecs_mask': batch_dicturb_kvecs_mask} # mask要用来算loss
        
        
        return data



@register_mol_graph_collator('atom_graph_cl') 
class AtomGraphCLCollator(object):
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.augmentor = BaseMolAugmentor(cfg.mol_graph_collator_cfg.atom_mask_rate, cfg.mol_graph_collator_cfg.bond_mask_rate)
    
    def augment_graphs(self, graphs):
        return [self.augmentor(graph) for graph in graphs]
    
    def __call__(self, item_list):
        graphs, kvecs_list = list(zip(*item_list))
        g1s, g2s = self.augment_graphs(graphs), self.augment_graphs(graphs)
        batched_g1 = dgl.batch(g1s)
        batched_g2 = dgl.batch(g2s)
        
        data = {}
        data['x1'] = {'atom_graph':batched_g1}
        data['x2'] = {'atom_graph':batched_g2}
        data['labels'] = None         
        return data


PolymerAtomGraphMAECollator = register_mol_graph_collator('polymer_atom_graph_mae')(AtomGraphMAECollator)  