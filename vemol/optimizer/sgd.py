import numpy as np
from dataclasses import dataclass, field
import torch.optim


from vemol.optimizer import register_optimizer
from vemol.optimizer.base_optimizer import OptimizerConfig, Optimizer
from hydra.core.config_store import ConfigStore

@dataclass
class SGDConfig(OptimizerConfig):
    optimizer: str = field(
        default='sgd', metadata={"help": "the optimizer method to train network)"}
    )
    
    momentum: float = field(
        default=0.9, metadata={"help": "momentum of sgd"}
    )
    
    lr: float = field(
        default=0.1, metadata={'help': "The initial learning rate (default: 0.1)."}
    )



@register_optimizer('sgd', SGDConfig)
class SGDOptimizer(Optimizer):
    def __init__(self, cfg: SGDConfig):
        super(SGDOptimizer, self).__init__(cfg)
        

    def _build_optimizer(self, model):
        """
        build sgd
        """
        params = self._get_model_grouped_parameters(model)
        self.optimizer = torch.optim.SGD(
            params, weight_decay=self.cfg.weight_decay, lr=self.cfg.lr, momentum=self.cfg.momentum
        )
        return self.optimizer
        
if __name__ == '__main__':
    cs = ConfigStore.instance()
    node = SGDConfig()
    name = 'sgd'
    node._name = name
    cs.store(name=name, group="optimizer", node=node)
    print(node)
    optimizer = SGDOptimizer(node)
    model = torchvision.models.resnet18(pretrained=False)
    optimizer.set_model(model)
    print(optimizer.get_optimizer())
