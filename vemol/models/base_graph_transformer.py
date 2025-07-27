from dataclasses import dataclass
from omegaconf import II
from typing import Dict

import dgl
from dgl import function as fn
from dgl.nn.functional import edge_softmax
import torch
import torch.nn as nn

from vemol.models import register_model
from vemol.models.base_model import ModelConfig, BaseModel
from vemol.models.base_model import base_model_wrapper, cl_model_wrapper

from vemol.modules.mlp import MLPModule
from vemol.modules.pooling_layer import FragmentPoolingReadOut

class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, dropout, activation_dropout):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ffn = MLPModule(d_in=d_model, d_out=d_model, d_hidden=d_model*4, 
                             n_layers=2, activation=nn.GELU(), dropout=activation_dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        res = x
        x = self.norm(x)
        x = self.ffn(x)
        x = self.dropout(x)
        out = x + res
        return out

class GraphSelfAttentionNetwork(nn.Module):
    def __init__(self, d_model, n_heads, dropout, attention_dropout):
        super().__init__()
        self.d_model = d_model
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.scale = d_model**(-0.5)
        self.n_heads = n_heads
        self.qkv = nn.Linear(d_model, d_model*3)
        self.out_fc = nn.Linear(d_model, d_model)

    def pretrans_edges(self, edges):
        edge_h = edges.src['hv']
        return {"he": edge_h}
    
    def forward(self, g: dgl.DGLGraph, node_feature: torch.Tensor):
        res = node_feature 
        g = g.local_var()
        dist_attn = g.edata['dist_attn']
        path_attn = g.edata['path_attn']
        node_feature = self.norm(node_feature)
        
        qkv = self.qkv(node_feature).reshape(-1, 3, self.n_heads, self.d_model // self.n_heads).permute(1, 0, 2, 3)
        q, k, v = qkv[0]*self.scale, qkv[1], qkv[2]
        g.dstdata.update({'K': k})
        g.srcdata.update({'Q': q})
        g.apply_edges(fn.u_dot_v('Q', 'K', 'node_attn'))

        g.edata['a'] = g.edata['node_attn'] + dist_attn.reshape(len(g.edata['node_attn']), -1, 1) + path_attn.reshape(len(g.edata['node_attn']), -1, 1) 
        g.edata['sa'] = self.attention_dropout(edge_softmax(g, g.edata['a']))
        
        g.ndata['hv'] = v.view(-1, self.d_model)
        g.apply_edges(self.pretrans_edges)
        g.edata['he'] = ((g.edata['he'].view(-1, self.n_heads, self.d_model//self.n_heads)) * g.edata['sa']).view(-1, self.d_model)
        
        g.update_all(fn.copy_e('he', 'm'), fn.sum('m', 'agg_h'))
        x = self.out_fc(g.ndata['agg_h'])
        x = self.dropout(x)
        out = res + x
        return out


@dataclass
class BaseGraphTransformerConfig(ModelConfig):
    model: str = "base_graph_transformer"
    input_dim: int = -1 # 137
    output_dim: int = -1 # 
    d_model: int = 512
    num_layers: int = 6
    dropout: float = 0.1
    activation_dropout: float = 0.1
    attention_dropout: float = 0.1
    n_heads: int = 8
    readout: str = 'mean'
    max_length: int = II("dataset.max_length")
    fragment_pool: bool = False
    num_predictor_layers: int = 2
    

class GraphTransformerLayer(nn.Module):
    def __init__(self, cfg: BaseGraphTransformerConfig):
        super().__init__()
        self.attn = GraphSelfAttentionNetwork(cfg.d_model, cfg.n_heads, cfg.dropout, cfg.attention_dropout)
        self.ffn = FeedForwardNetwork(cfg.d_model, cfg.dropout, cfg.activation_dropout)

    def forward(self, g: dgl.DGLGraph, node_feature: torch.Tensor):
        node_feature = self.attn(g, node_feature)
        node_feature = self.ffn(node_feature)
        return node_feature


class PathAttentionScore(nn.Module):
    def __init__(self, hidden_size=128, max_length=3, head=1) -> None:
        super().__init__()
        self.max_length = max_length
        self.head = head
        self.trip_fortrans = nn.ModuleList([
          nn.Linear(hidden_size, 1, bias=False) for _ in range(max_length)
        ])
        
    def forward(self, paths: torch.Tensor, node_feature: torch.Tensor):
        # path: E*max_length 
        paths[paths<0] = -1
        # paths[paths>=node_feature.shape[0]] = -1
        attn_scores = []
        # print(node_feature.shape)
        for i in range(self.max_length):
            idxs = paths[:, i]
            # idxs = torch.clip(idxs, min=-1, max=node_feature.shape[0]) 
            s = torch.cat([self.trip_fortrans[i](node_feature), torch.zeros(size=(1, self.head)).to(node_feature)], dim=0)[idxs]
            attn_scores.append(s)
        path_length = torch.sum(paths>=0, dim=-1, keepdim=True).clip(min=1)
        attn_score = torch.sum(torch.stack(attn_scores, dim=-1), dim=-1)
        attn_score = attn_score/path_length
        return attn_score 
    
# @register_model("base_graph_transformer", dataclass=BaseGraphTransformerConfig)
# @base_model_wrapper
class BaseGraphTransformer(nn.Module):
    def __init__(self, cfg: BaseGraphTransformerConfig):
        super().__init__()
        # print(cfg)
        self.cfg = cfg
        self.distance_embedding = nn.Embedding(cfg.max_length, 1)
        nn.init.constant_(self.distance_embedding.weight, 0)
        self.path_embedding = PathAttentionScore(cfg.d_model, max_length=cfg.max_length)
        
        self.proj_in = nn.Linear(cfg.input_dim, cfg.d_model)
        self.layers = nn.ModuleList([
            GraphTransformerLayer(cfg) for _ in range(cfg.num_layers)
        ])
        self.readout = cfg.readout
        if cfg.fragment_pool:
            self.fragment_pooler = FragmentPoolingReadOut(cfg.d_model, readout=cfg.readout)
        if cfg.output_dim<0:
            cfg.output_dim = cfg.d_model
        # self.predictor = nn.Linear(cfg.d_model, cfg.output_dim)
        self.predictor = MLPModule(d_in=cfg.d_model, d_out=cfg.output_dim, d_hidden=cfg.d_model, n_layers=cfg.num_predictor_layers, activation=nn.GELU(), dropout=cfg.dropout)
    
    def get_init_node_feature(self, data, graph_key='atom_graph', node_key='h'):
        g = data[graph_key]
        return self.proj_in(g.ndata[node_key])
    
    def get_init_attn(self, data: Dict, node_feature, graph_key='atom_graph', distance_key='distance', path_key='paths'):
        g = data[graph_key]
        distance = g.edata[distance_key]
        dist_attn = self.distance_embedding(distance)
        paths = g.edata[path_key] 
        path_attn = self.path_embedding(paths, node_feature)
        return dist_attn, path_attn
    
    def get_output(self, data, node_feature):
        g = data['atom_graph']
        if self.readout != 'none':
            if self.cfg.fragment_pool:
                x = self.fragment_pooler(data, node_feature)
            else:
                g.ndata['_pool_h'] = node_feature
                x = dgl.readout_nodes(g, '_pool_h', op=self.readout)
        else:
            x = node_feature
        return x
    
    def forward_to_embedding(self, data):
        return self.forward_to_graph_embedding(data)
    
    def forward_to_graph_embedding(self, data):
        g = data['atom_graph']
        node_feature = self.get_init_node_feature(data)
        dist_attn, path_attn = self.get_init_attn(data, node_feature)
        g.edata['dist_attn'] = dist_attn
        g.edata['path_attn'] = path_attn
        
        for i, layer in enumerate(self.layers):
            node_feature = layer(g, node_feature)

        out = self.get_output(data, node_feature)
        return out 
        
    def forward(self, data):
        # 需要额外提供的字段: distance, paths
        x = self.forward_to_graph_embedding(data)
        out = self.predictor(x)
        return out
    
WrappedBaseGraphTransformer = base_model_wrapper(BaseGraphTransformer)
register_model("base_graph_transformer", dataclass=BaseGraphTransformerConfig)(WrappedBaseGraphTransformer)

# 用于对比学习
@dataclass
class CLGraphTransformerConfig(BaseGraphTransformerConfig):
    model: str = "cl_graph_transformer"
    
WrappedCLGraphTransformer = base_model_wrapper(cl_model_wrapper(BaseGraphTransformer))
register_model("cl_graph_transformer", dataclass=CLGraphTransformerConfig)(WrappedCLGraphTransformer)

if __name__ == "__main__":
    pass 

