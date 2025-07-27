from dataclasses import dataclass, field
from vemol.criterion import register_criterion
from vemol.criterion.base_criterion import Criterion, CriterionCfg
from hydra.core.config_store import ConfigStore
import torch
import numpy as np

@dataclass
class CrossEntropyMAECfg(CriterionCfg):
    criterion: str = field(
        default='cross_entropy_mae', metadata={'help': "The name of criterion."}
    )
    valid_criterion: str = "acc"

@register_criterion('cross_entropy_mae', CrossEntropyMAECfg)
class CrossEntropyMAE(Criterion):
    def __init__(self, cfg: CrossEntropyMAECfg):
        super(CrossEntropyMAE, self).__init__(cfg)
        self.criterion = torch.nn.CrossEntropyLoss(reduction='none')

    def _forward(self, predict, labels):
        result = {}
        
        mask = labels['atom_mask']
        labels = labels['labels'].long().reshape(-1)
        # print(predict.shape, labels.shape, mask.shape);exit()
        pred_loss = self.criterion(predict, labels)
        sample_size = mask.sum()   #  KPGT是预测全部, 不仅包括mask的部分, 还有不mask的部分
        pred_loss = (pred_loss*mask).sum() / sample_size
        result['loss'] = pred_loss
        result['sample_size'] = sample_size
        result['_predict'] = [0]
        result['log'] = {
            'loss': result['loss'].item(),
            'sample_size': int(sample_size.item()),
            '_label': [0],
            '_predict': [0],
        }
        return result
        
        
if __name__ == '__main__':
    cs = ConfigStore.instance()
    node = CrossEntropyMAECfg()
    name = 'cross_entropy'
    node._name = name
    cs.store(name=name, group="criterion", node=node)
    print(node)
    cs = CrossEntropyMAE(node)
    print(cs)
    x = torch.randn(10, 4)
    y = torch.randint(low=0, high=4, size=(10,))
    mask = torch.randint(low=0, high=2, size=(10, ))
    samples = {'labels': y, 'atom_mask': mask}
    print(cs(x, samples))