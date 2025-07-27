from dataclasses import dataclass, field
from vemol.criterion import register_criterion
from vemol.criterion.base_criterion import Criterion, CriterionCfg
from hydra.core.config_store import ConfigStore
import torch
import numpy as np

@dataclass
class BinaryCrossEntropyCfg(CriterionCfg):
    criterion: str = field(
        default='binary_cross_entropy', metadata={'help': "The name of criterion."}
    )
    valid_criterion: str = "rocauc"

@register_criterion('binary_cross_entropy', BinaryCrossEntropyCfg)
class BinaryCrossEntropy(Criterion):
    def __init__(self, cfg: BinaryCrossEntropyCfg):
        super(BinaryCrossEntropy, self).__init__(cfg)
        self.criterion = torch.nn.BCEWithLogitsLoss(reduction='none')

    def _forward(self, predict, labels):
        result = {}
        
        is_labeled = (~torch.isnan(labels)).to(torch.float32)
        labels = torch.nan_to_num(labels, nan=0.0)
        sample_size = is_labeled.sum()
        pred_loss = (self.criterion(predict, labels) * is_labeled).sum() / sample_size
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
    node = BinaryCrossEntropyCfg()
    name = 'binary_cross_entropy'
    node._name = name
    cs.store(name=name, group="criterion", node=node)
    print(node)
    cs = BinaryCrossEntropy(node)
    print(cs)
    x = torch.randn(10, 4)
    y = torch.randint(low=0, high=1, size=(10, 4), dtype=torch.float32)
    samples = {'labels': y}
    print(cs(x, samples))