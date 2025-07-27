from typing import List

import dgl 
import torch 
import torch.nn as nn 

from vemol.chem_utils.fingerprint.fingerprint import FP_DIM

class KnowledgePooling(nn.Module):
    def __init__(self, d_model, fp_name='ecfp', readout='sum'):
        super().__init__()
        # 可以变为multi-head
        self.d_model = d_model
        self.linear_k = nn.Linear(d_model, d_model, bias=False)
        self.linear_q = nn.Linear(FP_DIM[fp_name], d_model, bias=False)
        self.linear_v = nn.Linear(d_model, d_model)
        self.readout = readout
        self.gru = nn.GRUCell(d_model, d_model)
        
    def forward(self, bg: dgl.DGLGraph, node_feature: torch.Tensor, k_feature: torch.Tensor):
        with bg.local_scope():
            k_feature = self.linear_q(k_feature)
            # return k_feature
            q = dgl.broadcast_nodes(bg, k_feature)
            k = self.linear_k(node_feature/self.d_model**0.5)
            v = self.linear_v(node_feature)
            # print(k.shape, q.shape)
            score = (k*q).sum(-1)
            bg.ndata['score'] = score
            # 在每个子图内对每个节点上的标量做softmax
            bg.ndata['attn'] = dgl.softmax_nodes(bg, 'score')
            bg.ndata['_kpool_h'] = bg.ndata['attn'].unsqueeze(-1)*v
            out = dgl.readout_nodes(bg, '_kpool_h', op=self.readout) 
            # 前边已经softmax了，这里使用sum pooling好一些
            # 这个gru感觉有点没必要, 用残差连接就行
            return self.gru(out, k_feature)
            # return out + k_feature





