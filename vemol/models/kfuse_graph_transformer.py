from dataclasses import dataclass
from omegaconf import II
from typing import Dict, List

import dgl
from dgl import function as fn
from dgl.nn.functional import edge_softmax
import torch
import torch.nn as nn

from vemol.models import register_model
from vemol.models.base_model import ModelConfig, BaseModel
from vemol.models.base_model import base_model_wrapper, cl_model_wrapper

from vemol.modules.mlp import MLPModule

from vemol.models.base_graph_transformer import BaseGraphTransformerConfig, BaseGraphTransformer
from vemol.modules.knowledge_fusion_layer import KNOWLEDGE_FUSIONS

@dataclass
class KfuseGraphTransformerConfig(BaseGraphTransformerConfig):
    model: str = "kfuse_graph_transformer"
    kvec_names: List[str] = II("dataset.kvec_names")
    fusion: str = "cross_attention"  # or gru


class KfuseGraphTransformer(BaseGraphTransformer):
    def __init__(self, cfg: KfuseGraphTransformerConfig):
        super().__init__(cfg)

        if len(cfg.kvec_names) > 0:
            self.k_fusion = KNOWLEDGE_FUSIONS[cfg.fusion](cfg.d_model, cfg.kvec_names)
    
    def get_output(self, data, node_feature):
        node_feature = self.k_fusion(data['atom_graph'], node_feature, data['kvecs'])
        return super().get_output(data, node_feature)
    
    def forward_to_embedding(self, data):
        return self.forward_to_graph_embedding(data)
        
    # def forward(self, data):
    #     # 需要额外提供的字段: distance, paths
    #     x = self.forward_to_graph_embedding(data)
    #     out = self.predictor(x)
    #     return out

WrappedKfuseGraphTransformer = base_model_wrapper(KfuseGraphTransformer)
register_model("kfuse_graph_transformer", dataclass=KfuseGraphTransformerConfig)(WrappedKfuseGraphTransformer)

if __name__ == "__main__":
    pass 

