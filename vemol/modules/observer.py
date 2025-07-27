import torch 
import torch.nn as nn

__all__ = ['Observer']


class Observer(nn.Module):
    # for monitoring, visualization and debugging
    # name参数可以用于实现不同观测点观察不同的指标
    def __init__(self, name=None):
        super().__init__()
        self.name = name if name is not None else ""
        if name is not None:
            self.__class__.__name__ = name

    def forward(self, x):
        return x


if __name__ == "__main__":
    obs = Observer("WJX")
    from Taiyi.utils.regisiter import Regisiter 
    from Taiyi.extensions.backward_extension.backward_output_extension import BackwardOutputExtension 
    Regisiter.register_backward(obs, [BackwardOutputExtension()])
    x = torch.randn(2, 3, requires_grad=True)
    # x = torch.randn(2, 3, requires_grad=False)
    y = obs(x)
    z = y.sum()
    z.backward()
    print(x.grad)
    print(obs.output_grad) 
    # observer的输入要求是需要梯度的tensor!
    # Taiyi 遇到输入不需要梯度的也不会报错, 但是不会生成数据供计算模块计算, 下次可以问问junlong