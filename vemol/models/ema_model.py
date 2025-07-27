import copy 
from typing import Optional 

import torch.nn as nn
from vemol.models.base_gnn import BaseModel

import logging

logger = logging.getLogger(__name__)

class EMAModel(BaseModel):
    # wrapper for ema model
    # 暂时不支持多GPU
    def __init__(self, model:BaseModel, decay=0.5):
        super().__init__(model.cfg)
        self.model = copy.deepcopy(model.get_model()) # ema_model
        self.model.eval()
        self.decay = decay
        for p in self.model.parameters():
            p.requires_grad_(False)

    def update_ema(self, new_model: BaseModel):
        ema_model_dict = self.model.state_dict()
        new_model_dict = new_model.state_dict()
        for k, v in new_model_dict.items():
            if k in ema_model_dict:
                ema_model_dict[k].data.copy_(
                    self.decay * ema_model_dict[k].data + (1.0 - self.decay) * v.data)
            else:
                raise ValueError(f"Key {k} is not in the ema model's state_dict")
        self.model.load_state_dict(ema_model_dict)


def build_ema_model(model: BaseModel, decay) -> Optional[EMAModel]:
    if decay<0:
        return None 
    else:
        return EMAModel(model, decay)