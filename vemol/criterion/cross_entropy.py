from dataclasses import dataclass, field
from vemol.criterion import register_criterion
from vemol.criterion.base_criterion import Criterion, CriterionCfg
from hydra.core.config_store import ConfigStore
import torch
import numpy as np

@dataclass
class CrossEntropyCfg(CriterionCfg):
    criterion: str = field(
        default='cross_entropy', metadata={'help': "The name of criterion."}
    )
    valid_criterion: str = "acc"

@register_criterion('cross_entropy', CrossEntropyCfg)
class CrossEntropy(Criterion):
    def __init__(self, cfg: CrossEntropyCfg):
        super(CrossEntropy, self).__init__(cfg)
        self.criterion = torch.nn.CrossEntropyLoss(reduction='mean')

    def _forward(self, predict, labels):
        result = {}
        sample_size = 1
        labels = labels.long().reshape(-1)
        pred_loss = self.criterion(predict, labels)
        result['loss'] = pred_loss
        result['sample_size'] = sample_size
        result['_predict'] = [0]
        result['log'] = {
            'loss': result['loss'].item(),
            'sample_size': sample_size,
            '_label': [0],
            '_predict': [0],
        }
        return result
        
        
if __name__ == '__main__':
    cs = ConfigStore.instance()
    node = CrossEntropyCfg()
    name = 'cross_entropy'
    node._name = name
    cs.store(name=name, group="criterion", node=node)
    print(node)
    cs = CrossEntropy(node)
    print(cs)
    x = torch.randn(10, 4)
    y = torch.randint(low=0, high=4, size=(10,))
    
    samples = {'labels': y}
    print(cs(x, samples))