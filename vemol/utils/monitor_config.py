
import torch.nn as nn
# from vemol.modules.norm_layers.batchnorm import RegBatchNorm1d


MONITOR_DEMO_1D_CONFIG = {
    'Linear': [['WeightNorm', 'linear(10, 0)'], ['OutputGradSndNorm', 'linear(10, 0)']],
    # 'Observer': [['InputSndNorm', 'linear(10, 0)'], ['OutputGradSndNorm', 'linear(10, 0)']]
    # 'Observer': [['InputSndNorm', 'linear(10, 0)']]
    # 'Observer': [['OutputGradSndNorm', 'linear(10, 0)']],
    # "nn.BatchNorm1d": [['MeanTID', 'linear(10, 0)']],
    # nn.BatchNorm1d: [['MeanTID', 'linear(10, 0)']],
    'BatchNorm1d': [['MeanTID', 'linear(10, 0)'],
                    ['VarTID', 'linear(10, 0)'],
                    ['MeanTIDRatio', 'linear(10, 0)'],
                    ['VarTIDRatio', 'linear(10, 0)'],], # 只能这么写, hydra不支持class作为dict的key
    # 'Linear': [['MeanTID', 'linear(10, 0)']],
    # 'Linear': [['WeightNorm', 'linear(10, 0)']],
}

MONITOR_MPP_GT_CONFIG = {
    # 'Linear': [['WeightNorm', 'linear(10, 0)'], ['OutputGradSndNorm', 'linear(10, 0)']],
    # 'Observer': [['InputSndNorm', 'linear(10, 0)'], ['OutputGradSndNorm', 'linear(10, 0)']]
    # 'Observer': [['InputSndNorm', 'linear(10, 0)']]
    # RegBatchNorm1d: [['MeanTID', 'linear(10, 0)'], ['VarTID', 'linear(10, 0)']],
    'BatchNorm1d': [['MeanTID', 'linear(10, 0)'], 
                    ['VarTID', 'linear(10, 0)'],
                    ['MeanTIDRatio', 'linear(10, 0)'],
                    ['VarTIDRatio', 'linear(10, 0)'],], #
}
