from dataclasses import dataclass, field
from vemol.metric import register_metric
from vemol.metric.base_metric import Metric, MetricCfg
from hydra.core.config_store import ConfigStore
import torch
import numpy as np

from vemol.metric.__bin_evaluator import Evaluator

@dataclass
class AccCfg(MetricCfg):
    metric: str = field(
        default='acc', metadata={'help': "The name of metric."}
    )

@register_metric('acc', AccCfg)
class Acc(Metric):
    def __init__(self, cfg: AccCfg):
        super(Acc, self).__init__(cfg)

    def _forward(self, predicts, labels):
        acc = (predicts == labels).sum().item() / len(labels) # bool tensor不能用.mean(), 只能用sum()
        score = {'acc': acc}
        return score
        
        
if __name__ == '__main__':
    cs = ConfigStore.instance()
    node = BinaryGraphClsCfg()
    name = 'binary_graph_cls'
    node._name = name
    cs.store(name=name, group="metric", node=node)
    print(node)
    cs = BinaryGraphCls(node)
    print(cs)
    size = (100, 4)
    x = torch.randint(low=0, high=2, size=size)
    y = x - 0.5 + torch.randn(*size)
    y = torch.sigmoid(y)
    # print(x, y, sep='\n')
    print(cs(x, y))