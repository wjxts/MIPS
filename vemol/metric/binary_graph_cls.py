from dataclasses import dataclass, field
from vemol.metric import register_metric
from vemol.metric.base_metric import Metric, MetricCfg
from hydra.core.config_store import ConfigStore
import torch
import numpy as np

from vemol.metric.__bin_evaluator import Evaluator

@dataclass
class BinaryGraphClsCfg(MetricCfg):
    metric: str = field(
        default='binary_graph_cls', metadata={'help': "The name of metric."}
    )

@register_metric('binary_graph_cls', BinaryGraphClsCfg)
class BinaryGraphCls(Metric):
    def __init__(self, cfg: BinaryGraphClsCfg):
        super(BinaryGraphCls, self).__init__(cfg)
        metrics = ['rocauc', 'ap', 'acc', 'f1']
        self.evaluators = {metric: Evaluator(eval_metric=metric) for metric in metrics}

    def _forward(self, predicts, labels):
        # 注意evalutaor的输入是(labels, predicts), 接口定义是反的
        score = {metric: evaluator.eval(labels, predicts) for metric, evaluator in self.evaluators.items()}
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