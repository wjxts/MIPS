import torch.distributed as dist
import torch
from torch.nn.parallel import DistributedDataParallel


def apply_to_sample(f, sample):

    if hasattr(sample, '__len__') and len(sample) == 0:
        return {}

    def _apply(x):
        if torch.is_tensor(x):
            return f(x)
        elif isinstance(x, dict):
            return {key: _apply(value) for key, value in x.items()}
        elif isinstance(x, list):
            return [_apply(x) for x in x]
        elif isinstance(x, tuple):
            return tuple(_apply(x) for x in x)
        elif isinstance(x, set):
            return {_apply(x) for x in x}
        else:
            return x

    return _apply(sample)


def mean_reduce(sample, device_count=2, device=0):
    def _mean_reduce(tensor):
        if torch.is_tensor(tensor) is not True:
            tensor = torch.tensor(tensor)
        dist.reduce(tensor, device)
        return tensor / device_count
    return apply_to_sample(_mean_reduce, sample)



def get_model(model):
    if isinstance(model, DistributedDataParallel):
        return model.module
    else:
        return model