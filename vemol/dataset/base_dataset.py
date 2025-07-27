import os
from builtins import str

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from dataclasses import field
from omegaconf import II, MISSING
from dataclasses import dataclass

from abc import ABC,abstractmethod 
from typing import Literal

from vemol.dataclass import BaseDataclass

import logging 

logger = logging.getLogger(__name__)

@dataclass
class BaseDatasetConfig(BaseDataclass):
    dataset: str = field(
        default=None, metadata={'help': "The class of dataset."}
    )
    name: str = field(
        default='', metadata={'help': "The name of dataset."}
    )
    data_parallel_size: int = II("common.data_parallel_size")
    validate: bool = II("common.validate")
    device: int = II("common.device")
    num_workers: int = 0
    batch_size: int = 32
    valid_batch_size: int = II("dataset.batch_size")
    input_dim: int = -1 # 
    output_dim: int = -1 # 
    hidden_size: int = -1
    
# not useful currently
class BaseDataLoader(ABC):
    def __init__(self, cfg):
        self.cfg = cfg
        self.dataset = {}
        self.dataset['train'] = self._get_dataset('train')
        self.num_batches_per_epoch = len(self.dataset['train']) // self.cfg.batch_size
        if cfg.validate:
            self.dataset['valid'] = self._get_dataset('valid')
            self.dataset['test'] = self._get_dataset('test')
        # print(f"train: {len(self.dataset['train'])}, valid: {len(self.dataset['valid'])}, test: {len(self.dataset['test'])}");exit()
    
    @abstractmethod
    def _get_dataset(self, split=Literal['train', 'valid', 'test']) -> Dataset:
        pass 
    
    def get_dataloader(self, split='train', epoch=0):
        dataset = self.dataset[split]
        if self.cfg.data_parallel_size > 1:
            sampler = DistributedSampler(dataset)
            assert self.cfg.batch_size % self.cfg.data_parallel_size == 0, "batch size should be divided by data_parallel_size"
            batch_size = self.cfg.batch_size // self.cfg.data_parallel_size
            train_shuffle = False
        else:
            sampler = None
            batch_size = self.cfg.batch_size
            train_shuffle = True
        '''
        collate_fn = None的行为是: 
        如果样本是张量 (torch.Tensor): 将样本作为第 0 维沿着堆叠，生成一个形状更大的张量。
        例如，若输入 5 个形状是 (3, 224, 224) 的张量，结果会是一个 (5, 3, 224, 224) 的张量。
        如果样本是数字（标量，比如标记的标签）: 直接放入列表中。
        如果样本是可迭代对象（比如元组或字典）: collate_fn 会进行递归处理，并按照结构深度逐步处理每个元素，最终生成类似样本数据的结构。
        如果无法自动处理样本的结构，则会抛出错误。
        '''
        if split == 'train':
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=train_shuffle, sampler=sampler, num_workers=self.cfg.num_workers, collate_fn=dataset.collator, drop_last=True)
        else:
            dataloader = DataLoader(dataset, batch_size=self.cfg.valid_batch_size, shuffle=False, sampler=sampler, num_workers=self.cfg.num_workers, collate_fn=dataset.collator)
        if self.cfg.data_parallel_size > 1:
            dataloader.sampler.set_epoch(epoch)    
        logger.info(f"get {split} dataloader ({len(dataloader)}), batch size: {batch_size}, data_parallel_size: {self.cfg.data_parallel_size}, num_workers: {self.cfg.num_workers}")
        return dataloader



def standard_dataset_to_dataloader(dataset_cls):
    class Loader(BaseDataLoader):
        def __init__(self, cfg: BaseDatasetConfig):
            super(Loader, self).__init__(cfg)
            
        def _get_dataset(self, split=Literal['train', 'valid', 'test']) -> Dataset:
            return dataset_cls(self.cfg, split)
        
    return Loader