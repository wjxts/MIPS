from dataclasses import field
from typing import Any
from abc import ABC,abstractmethod 

from hydra.core.config_store import ConfigStore
import torch
from torch import nn, Tensor
from torch.nn.parallel import DistributedDataParallel as DDP

from dataclasses import dataclass
from omegaconf import II, MISSING

from vemol.dataclass import BaseDataclass
from vemol.models import register_model
from vemol.modules import get_dataclass

import logging

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig(BaseDataclass):       
    model: str = field(
        default='model', metadata={"help": "model name"}
    )
    device: int = II("common.device")
    data_parallel_size: int = II("common.data_parallel_size")
    input_dim: int = -1
    output_dim: int = -1 # do not set model.output_dim, it will be overrided by dataset.output_dim
    d_model: int = -1
    num_layers: int = -1
    # input_dim: int = II("dataset.input_dim") # 不是一开始决定的, 还可能在dataset.init里计算
    # output_dim: int = II("dataset.output_dim")
    @classmethod
    def get_module(cls, module):
        return get_dataclass(module)


class BaseModel(ABC):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.device = cfg.device
        self.model = None

    def state_dict(self):
        return self.model.state_dict()
    
    def load_state_dict(self, state_dict, strict=True):
        # 处理torch.load strict=False不支持size mismatch的问题
        if not strict:
            logger.info("="*120)
            keys = list(state_dict.keys())
            for k in keys:
                v = state_dict[k]
                if k not in self.model.state_dict():
                    logger.info(f"Unexpected Key: {k} is not in the model's state_dict")
                else:
                    # print(k, v.shape, self.model.state_dict()[k].shape)
                    if v.shape != self.model.state_dict()[k].shape:
                        logger.info(f"Key {k} is not the same shape as the model's state_dict, {v.shape} != {self.model.state_dict()[k].shape}, pop it.")
                        state_dict.pop(k)
                        if "predictor" not in k:
                            raise ValueError(f"{k} does not match the model's state_dict")
            for k, v in self.model.state_dict().items():
                if k not in state_dict:
                    logger.info(f"Missing Key: {k} is not in the state_dict")
            logger.info("="*120)
        return self.model.load_state_dict(state_dict, strict=strict)
    
    def get_model(self):
        return self.model
    
    def move_model_to_gpu(self):
        self.model = self.model.to(self.device)
        
    def wrap_ddp(self): # move_model_to_gpu和wrap_ddp是定义这个basemodel的原因
        # self.model = DDP(self.model, device_ids=[self.device])
        self.model = DDP(self.model, device_ids=[self.device], find_unused_parameters=True)
        # print(list(self.model.state_dict().keys()))[:2];exit() # 参数字段前边会多一个'module.'
    
    # 可以在子类中覆盖
    def train_step(self, sample):
        return self.forward(sample)
    
    # 可以在子类中覆盖
    @torch.no_grad() # 不一定非得加这个装饰器
    def valid_step(self, sample):
        return self.forward(sample)
    
    def train(self):
        self.model.train()
    
    def eval(self):
        self.model.eval()
    
    def parameters(self):
        return self.model.parameters()
    
    def named_parameters(self):
        return self.model.named_parameters()
    
    def modules(self):
        return self.model.modules()
    
    def named_modules(self):
        return self.model.named_modules()
    
    def forward(self, sample):
        return self.model(sample)
    
    def forward_to_embedding(self, sample):
        return self.model.forward_to_embedding(sample)
    # can replace many member function, to arrange
    # def __getattr__(self, name):
    #     return self.model.__getattribute__(name)
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    # does not work!
    def no_sync(self):
        self.model.no_sync()
        
    @property
    def training(self):
        return self.model.training
    
    

def base_model_wrapper(model_cls) -> BaseModel:
    # print("call model wrapper!")
    class WrappedModel(BaseModel):
        def __init__(self, cfg):
            super().__init__(cfg)
            self.model = model_cls(cfg)
    return WrappedModel

# 包装模型用于对比学习, 数据接口的规范是包含x1, x2两个字段

def cl_model_wrapper(model_cls) -> BaseModel:
    class WrappedCLModel(model_cls):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
        
        def forward(self, data):
            if 'x1' in data and 'x2' in data:
                return self.pretrain_forward(data)
            else:
                return self.finetune_forward(data)
        
        def pretrain_forward(self, data):
            data1 = data['x1']
            data2 = data['x2']
            # print(data1);exit()
            y1 = super().forward(data1)
            y2 = super().forward(data2)
            y = {'x1': y1, 'x2': y2}
            return y 
        
        def finetune_forward(self, data):
            y = super().forward(data)
            return y
    return WrappedCLModel


# class CLWrapper(nn.Module):
#     def __init__(self, model):
#         super().__init__()
#         self.model = model
    
#     def forward(self, data):
#         if 'x1' in data and 'x2' in data:
#             return self.pretrain_forward(data)
#         else:
#             return self.finetune_forward(data)
    
#     def pretrain_forward(self, data):
#         data1 = data['x1']
#         data2 = data['x2']
#         # print(data1);exit()
#         y1 = self.model(data1)
#         y2 = self.model(data2)
#         y = {'x1': y1, 'x2': y2}
#         return y 
    
#     def finetune_forward(self, data):
#         y = self.model(data)
#         return y