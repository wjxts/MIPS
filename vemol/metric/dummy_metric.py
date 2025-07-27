from dataclasses import dataclass, field
from vemol.metric import register_metric
from vemol.metric.base_metric import Metric, MetricCfg
from hydra.core.config_store import ConfigStore


@dataclass
class DummyCfg(MetricCfg):
    metric: str = field(
        default='dummy', metadata={'help': "The name of metric."}
    )

@register_metric('dummy', DummyCfg)
class DummyMetric(Metric):
    def __init__(self, cfg: DummyCfg):
        super(DummyMetric, self).__init__(cfg)
       
    def _forward(self, *args, **kwargs):
        raise NotImplementedError("This is a dummy metric, do not open validate mode.")
        
        
if __name__ == '__main__':
    cs = ConfigStore.instance()
    node = DummyCfg()
    # name = 'dummy'
    # node._name = name
    # cs.store(name=name, group="metric", node=node)
    print(node)
    cs = DummyMetric(node)
    labels = 1
    predicts = 2
    cs(labels, predicts)
    