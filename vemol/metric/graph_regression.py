from dataclasses import dataclass, field
from vemol.metric import register_metric
from vemol.metric.base_metric import Metric, MetricCfg
from hydra.core.config_store import ConfigStore
import torch
import numpy as np

from vemol.metric.__bin_evaluator import Evaluator

@dataclass
class GraphRegCfg(MetricCfg):
    metric: str = field(
        default='graph_reg', metadata={'help': "The name of metric."}
    )

@register_metric('graph_reg', GraphRegCfg)
class GraphReg(Metric):
    def __init__(self, cfg: GraphRegCfg):
        super(GraphReg, self).__init__(cfg)
        metrics = ['rmse', 'mae', 'r2']
        self.evaluators = {metric: Evaluator(eval_metric=metric) for metric in metrics}

    def _forward(self, predicts, labels):
        labels, mean, std = labels['labels'], labels['mean'], labels['std']
        # print(mean, std, sep='\n')
        # 注意evalutaor的输入是(labels, predicts), 接口定义是反的
        score = {metric: evaluator.eval(labels, predicts, mean=mean, std=std) for metric, evaluator in self.evaluators.items()}

        return score
        
        
if __name__ == '__main__':
    cs = ConfigStore.instance()
    node = GraphRegCfg()
    name = 'graph_reg'
    node._name = name
    cs.store(name=name, group="metric", node=node)
    print(node)
    cs = GraphReg(node)
    print(cs)
    size = (100, 4)
    labels = torch.randn(*size)
    predicts = labels + torch.randn(*size)
    # print(x, y, sep='\n')
    print(cs(predicts, labels))