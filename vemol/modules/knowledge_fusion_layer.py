from typing import List, Dict

import dgl 
import torch 
import torch.nn as nn 

from vemol.chem_utils.fingerprint.fingerprint import FP_DIM


KNOWLEDGE_FUSIONS = {}

def register_knowledge_fusion(name):
    def decorator(knowledge_fusion_module):
        KNOWLEDGE_FUSIONS[name] = knowledge_fusion_module
        return knowledge_fusion_module
    return decorator 

@register_knowledge_fusion('cross_attention')
class AtttentiveFusion(nn.Module):
    # 感觉融合knowledge不简单, 需要考虑到不同的knowledge之间的关系
    def __init__(self, d_model: int, knodes: List[str]):
        super().__init__()
        self.d_model = d_model
        self.knodes = knodes
        self.d_attn = d_model // 4
        self.k_proj = nn.ModuleDict([
            (k, nn.Linear(FP_DIM[k], self.d_attn, bias=False)) for k in knodes
        ])
        
        self.v_proj = nn.ModuleDict([
            (k, nn.Linear(FP_DIM[k], d_model)) for k in knodes
        ])
        
        self.Wq = nn.Linear(d_model, self.d_attn, bias=False)
    
    def build_vectors(self, bg, knodes, proj):
        # knodes: a dict of knowledge nodes
        vectors = []
        for fp_name, v in knodes.items():
            feature = proj[fp_name](v) # bs*d
            feature = dgl.broadcast_nodes(bg, feature) # N*d
            vectors.append(feature)
        vectors = torch.stack(vectors, dim=1) # N*k*d
        return vectors
    
    def forward(self, bg, x, knodes: Dict):
        # knodes: a dict of knowledge nodes {fp_name: bs*fp_dim}
        # x: N*d
        if len(knodes) == 0:
            return x
        k_vectors = self.build_vectors(bg, knodes, self.k_proj)  # N*k*d_attn
        v_vectors = self.build_vectors(bg, knodes, self.v_proj)  # N*k*d
        # print(x.shape, k_vectors.shape, v_vectors.shape);exit() #torch.Size([231, 128]) torch.Size([231, 6, 128]) torch.Size([231, 6, 128])
        # x: N*d, k_vector: N*k*d
        q = self.Wq(x) / (self.d_model**0.5) # N*d_attn
        q = q.unsqueeze(1) # N*1*d_attn
        
        attn = torch.bmm(q, k_vectors.transpose(1, 2)) # N*1*k
        attn = torch.softmax(attn, dim=-1) # N*1*k
        
        out = torch.bmm(attn, v_vectors).squeeze(1) # N*d
        return x + out*0.5
    

class GRUFusionLayer(nn.Module):
    def __init__(self, d_model, fp_name='ecfp'):
        super().__init__()
        self.d_model = d_model
        self.proj_knowledge = nn.Linear(FP_DIM[fp_name], d_model, bias=False)
        self.gru = nn.GRUCell(d_model, d_model)
        
    def forward(self, bg, node_feature, k_feature):
        k_feature = self.proj_knowledge(k_feature)
        k_feature = dgl.broadcast_nodes(bg, k_feature)
        node_feature = self.gru(k_feature, node_feature)
        return node_feature

@register_knowledge_fusion('gru')
class GRUFusion(nn.Module):
    def __init__(self, d_model: int, knodes: List[str]):
        super().__init__()
        self.d_model = d_model
        self.knodes = knodes
        self.k_fusions = nn.ModuleList([GRUFusionLayer(d_model, fp_name=fp_name) for fp_name in knodes])
    
    def forward(self, bg, x, knodes:  dict):
        if len(knodes) == 0:
            return x
        for i, fp_name in enumerate(self.knodes):
            k_feature = knodes[fp_name] 
            x = self.k_fusions[i](bg, x, k_feature)
        return x 


if __name__ == "__main__":
    pass 