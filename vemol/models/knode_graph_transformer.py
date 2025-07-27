from dataclasses import dataclass
from omegaconf import II
from typing import List 

import dgl
from dgl import function as fn
from dgl.nn.functional import edge_softmax
import numpy as np 
import torch
import torch.nn as nn

from vemol.models import register_model
from vemol.models.base_model import ModelConfig, BaseModel
from vemol.models.base_model import base_model_wrapper, cl_model_wrapper

from vemol.models.base_graph_transformer import BaseGraphTransformerConfig, BaseGraphTransformer
from vemol.models.knode_gnn import KnodeGNN

from vemol.chem_utils.fingerprint.fingerprint import FP_DIM

@dataclass
class KvecGraphTransformerConfig(BaseGraphTransformerConfig):
    model: str = "knode_graph_transformer"
    kvec_names: List[str] = II("dataset.kvec_names")


class KvecGraphTransformer(BaseGraphTransformer):
    def __init__(self, cfg: KvecGraphTransformerConfig):
        BaseGraphTransformer.__init__(self, cfg)
        KnodeGNN.init_kvec_params(self)
        self.virtual_path_distance_embedding = nn.Embedding(1, 1)
        nn.init.constant_(self.virtual_path_distance_embedding.weight, 0)
        self.kvec_predictors = nn.ModuleDict({
           name: nn.Linear(self.cfg.d_model, FP_DIM[name]) for name in self.kvec_names
        })
        if self.readout != 'none':
            self.predictor = nn.Linear((1+len(self.kvec_names))*self.cfg.d_model, self.cfg.output_dim)
        
    def get_init_node_feature(self, data):
        return KnodeGNN.get_init_node_feature(self, data)
    
    def get_init_attn(self, data, node_feature):
        dist_attn, path_attn = BaseGraphTransformer.get_init_attn(self, data, node_feature)
        # 区别是针对virtual_path有一个特殊的distance_embedding
        g = data['atom_graph']
        virtual_path = g.edata['virtual_path']
        dist_attn[virtual_path==1] = self.virtual_path_distance_embedding.weight
        return dist_attn, path_attn
    
    def forward(self, data):
        x = self.forward_to_graph_embedding(data)
        return x
    
    def get_output(self, data, node_feature):
        g = data['atom_graph']
        g.ndata['out'] = node_feature
        node_indicator = g.ndata['node_indicator']
        
        if self.readout != 'none': # indicate fine-tuning
            kvecs = [node_feature[node_indicator==i] for i in range(1, len(self.kvec_names)+1)]
            # print([x.shape for x in kvecs])
            g.remove_nodes(np.where(node_indicator.detach().cpu().numpy()>=1)[0])
            g_readout = dgl.readout_nodes(g, 'out', op=self.readout)
            feature_list = kvecs + [g_readout]
            out_feature = torch.cat(feature_list, dim=-1)
            out = self.predictor(out_feature)
        else: # indicate pre-training
            kvec_outs = {}
            for i, name in enumerate(self.kvec_names, start=1):
                kvec_outs[name] = self.kvec_predictors[name](node_feature[node_indicator==i])
            g_out = self.predictor(node_feature)
            out = {
                'node': g_out,
                'kvecs': kvec_outs
            }
        return out
        
        
        
        # readout = dgl.readout_nodes(g, 'out', op=self.readout)

WrappedKvecGraphTransformer = base_model_wrapper(KvecGraphTransformer)
register_model("knode_graph_transformer", dataclass=KvecGraphTransformerConfig)(WrappedKvecGraphTransformer)


if __name__ == "__main__":
    
    from vemol.chem_utils.smiles2graph.complete_graph import smiles2complete_graph
    from vemol.chem_utils.smiles2graph.knode_graph import add_knodes_to_complete_graph
    from vemol.chem_utils.featurizer.standard_featurizer import ATOM_FEATURE_DIM, BOND_FEATURE_DIM
    smiles = 'N1CCOCC1'
    # smiles = '[C]'
    complete_graph = smiles2complete_graph(smiles, max_length=3)
    kvecs = {'ecfp': torch.randn(1, 1024), 'md': torch.randn(1, 200), 'rdkfp': torch.randn(1, 1024)}
    knode_graph = add_knodes_to_complete_graph(complete_graph, kvecs, max_length=3)
    
    cfg = KvecGraphTransformerConfig()
    cfg.input_dim = ATOM_FEATURE_DIM
    cfg.output_dim = 1
    cfg.max_length = 3
    cfg.kvec_names = ['ecfp', 'md', 'rdkfp']
    model = KvecGraphTransformer(cfg)
    data = {
        'g': knode_graph,
        'kvecs': kvecs
    }
    y = model(data)
    print(y)
    
