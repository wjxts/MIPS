from typing import Dict

import dgl 
import torch
import torch.nn as nn 
import torch_scatter

from vemol.modules.gnn_layer import GNN_LAYERS

class GraphPooling(nn.Module):
    def __init__(self, readout='sum') -> None:
        super().__init__() 
        self.readout = readout
    
    def forward(self, node_feature: torch.Tensor, g: dgl.DGLGraph):
        g.ndata['_pool_h'] = node_feature # use a temporary key name
        x = dgl.readout_nodes(g, '_pool_h', op=self.readout)
        return x
    

class SubgraphPooling(nn.Module):
    def __init__(self, reduction='mean') -> None:
        super().__init__()
        self.reduction = reduction
        
    def forward(self, node_feature:torch.Tensor, batch_node_ids: torch.Tensor, batch_macro_node_ids: torch.Tensor):
        # batch_macro_node_ids: 1D-tensor
        # https://pytorch-scatter.readthedocs.io/en/latest/functions/scatter.html
        subgraph_embedding = torch_scatter.scatter(node_feature[batch_node_ids], batch_macro_node_ids, dim=0, reduce=self.reduction)
        return subgraph_embedding


class FragmentPoolingReadOut(nn.Module):
    def __init__(self, d_model, readout='mean'):
        super().__init__()
        self.d_model = d_model
        self.readout = readout
        self.frag_pooling = SubgraphPooling(reduction='mean')
        # layer_cls = GNN_LAYERS['gin']
        # self.gnn_layer = layer_cls(d_model, d_model)
    
    
    def forward(self, data: Dict, node_feature: torch.Tensor) -> torch.Tensor:
        fragment_feature = self.frag_pooling(node_feature, data['group_atom_idxs_1d'], data['macro_node_scatter_idxs'])
        self.fragment_feature = fragment_feature
        fragment_graph = data['fragment_graph']
        # fragment_feature = self.gnn_layer(fragment_graph, fragment_feature)
        fragment_graph.ndata['_pool_h'] = fragment_feature
        x = dgl.readout_nodes(fragment_graph, '_pool_h', op=self.readout)
        # atom_graph = data['atom_graph']
        # atom_graph.ndata['_pool_h'] = node_feature
        # x = dgl.readout_nodes(atom_graph, '_pool_h', op=self.readout)
        return x 
        