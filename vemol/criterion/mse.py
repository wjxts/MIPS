from dataclasses import dataclass, field
from vemol.criterion import register_criterion
from vemol.criterion.base_criterion import Criterion, CriterionCfg
from hydra.core.config_store import ConfigStore
import torch
import numpy as np

@dataclass
class MSECfg(CriterionCfg):
    criterion: str = field(
        default='mse', metadata={'help': "The name of criterion."}
    )
    valid_criterion: str = "rmse"

@register_criterion('mse', MSECfg)
class MSE(Criterion):
    def __init__(self, cfg: MSECfg):
        super(MSE, self).__init__(cfg)
        self.criterion = torch.nn.MSELoss(reduction='none')

    def _forward(self, predict, labels):
        result = {}
        labels, mean, std = labels['labels'], labels['mean'], labels['std']
        # print(predict.shape, labels.shape)
        # print(predict.sum(), labels.sum())
        mean, std = torch.from_numpy(mean).to(labels), torch.from_numpy(std).to(labels)
        is_labeled = (~torch.isnan(labels)).to(torch.float32)
        labels = torch.nan_to_num(labels, nan=0.0)
        labels = (labels - mean) / std
        sample_size = is_labeled.sum()
        pred_loss = (self.criterion(predict, labels) * is_labeled).sum() / sample_size
        # print("loss:", pred_loss)
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
    cfg = MSECfg()
    criterion = MSE(cfg)
    x = torch.randn(10, 4)
    y = torch.randx(10, 4)
    samples = {'labels': y}
    print(criterion(x, samples))