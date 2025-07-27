import contextlib
import json 
import logging 
from pathlib import Path 

import numpy as np
from typing import List, Dict
import torch 
import torch.nn as nn 

from omegaconf import DictConfig

logger = logging.getLogger(__name__)

METRIC_BEST_TYPE = {'rocauc': 'max', 'ap': 'max', 'acc': 'max', 'f1': 'max', 'rmse': 'min', 'mae': 'min', 'r2': 'max'}

# 这个函数与要探讨的问题相关, 可以修改

# RESULT_FILE_FUNC_DICT = {}

# def register_get_result_file_func(name):
#     def decorator(get_result_file_func):
#         RESULT_FILE_FUNC_DICT[name] = get_result_file_func
#         return get_result_file_func
#     return decorator 


def get_result_file(save_results_dir: str, dataset: str, scaffold:int, model: str, d_model:int, num_layers: int, ema_decay: float, seed: int) -> Path:
    base_result_dir = Path(save_results_dir)
    ema_str = f'_ema{ema_decay}' if ema_decay > 0 else ''
    return base_result_dir / f"{dataset}_scaffold{scaffold}_{model}{num_layers}_dmodel{d_model}{ema_str}_seed{seed}.json"   

def get_result_file_from_cfg(task: DictConfig) -> Path:
    base_result_dir = Path(task.common.save_results_dir)
    base_result_dir.mkdir(parents=True, exist_ok=True)
    target_str = "_" + "_".join(task.dataset.select_targets) if task.dataset.select_targets else ""
    file = get_result_file(save_results_dir=base_result_dir, 
                           dataset=f"{task.dataset.dataset}_{task.dataset.name}{target_str}", 
                           scaffold=task.dataset.scaffold_id,
                           model=task.model.model, 
                           d_model=task.model.d_model, 
                           num_layers=task.model.num_layers, 
                           ema_decay=task.common.ema_decay, 
                           seed=task.common.seed)
    return file

def empty_context():
    return contextlib.ExitStack()

def save_json(obj, file):
    # obj can be list or dict
    json.dump(obj, open(file, 'w'), indent=2)
    
def is_better(a, b, index):
    '''
    judge if a is better than b in the context of index
    '''
    greater_index = ['accuracy', 'acc', 'f1', 'precision', 'recall', 'ap', 'rocauc']
    less_index = ['loss', 'nll_loss', 'rmse', 'mae', 'mse', 'r2']
    if index in greater_index:
        return a > b
    elif index in less_index:
        return a < b
    else:
        raise ValueError('index {} is not supported'.format(index))

def merge_list(l: List):
    if len(l) == 0:
        return []
    if isinstance(l[0], torch.Tensor):
        return torch.cat(l, dim=0)
    elif isinstance(l[0], Dict):
        res = {k: torch.cat([d[k] for d in l], dim=0) for k in l[0].keys() if k not in ['mean', 'std']}
        if 'mean' in l[0]:
            res['mean'] = l[0]['mean']
            res['std'] = l[0]['std']
        return res # only support item = tensor or dict of tensor
    

def collect_test_results(train_score_list, valid_score_list, test_score_list,
                         ema_valid_score_list=None, ema_test_score_list=None,):
    d_result = {}
    for metric in valid_score_list[0]:
        # train_scores = [score[metric] for score in train_score_list]
        valid_scores = [score[metric] for score in valid_score_list]
        test_scores = [score[metric] for score in test_score_list]
        if ema_valid_score_list is not None:
            ema_valid_scores = [score[metric] for score in ema_valid_score_list]
            ema_test_scores = [score[metric] for score in ema_test_score_list]
        
        if METRIC_BEST_TYPE[metric] == 'max':
            arg_func = np.argmax
            extremal_func = max 
            extremal_name = "max"
        else:
            arg_func = np.argmin
            extremal_func = min
            extremal_name = "min"
        
        extremal_index = arg_func(valid_scores)
        # d_result[f'max_train_{metric}'] = max(train_scores)
        d_result[f'{extremal_name}_valid_{metric}'] = extremal_func(valid_scores)
        d_result[f'{extremal_name}_test_{metric}'] = extremal_func(test_scores)
        d_result[f'final_test_{metric}'] = test_scores[extremal_index]
        
        logger.info((
            # f"max train {metric}: {max(train_scores):.3f}, "
            f"{extremal_name} valid {metric}: {extremal_func(valid_scores):.3f}, "
            f"final test {metric}: {test_scores[extremal_index]:.3f}, "
            f"{extremal_name} test {metric}: {extremal_func(test_scores):.3f}"
            ))
        if ema_valid_score_list is not None:
            ema_extremal_index = arg_func(ema_valid_scores)
            d_result[f'{extremal_name}_ema_valid_{metric}'] = extremal_func(ema_valid_scores)
            d_result[f'{extremal_name}_ema_test_{metric}'] = extremal_func(ema_test_scores)
            d_result[f'final_ema_test_{metric}'] = ema_test_scores[ema_extremal_index]
            logger.info((
                f"{extremal_name} ema valid {metric}: {extremal_func(ema_valid_scores):.3f}, "
                f"final ema test {metric}: {ema_test_scores[ema_extremal_index]:.3f}, "
                f"{extremal_name} ema test {metric}: {extremal_func(ema_test_scores):.3f}"
                ))
        
    return d_result

def cal_grad_norm(model):
    total_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:  # Only consider parameters with gradients
            param_norm = param.grad.data.norm(2)  # L2 norm of the gradients
            total_norm += param_norm.item() ** 2  # Accumulate squared norm

    total_norm = total_norm ** 0.5  # Take the square root to get the total norm
    return total_norm
        
        
def get_n_model_params(model: nn.Module) -> int:
    model_parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    n_params = sum([np.prod(p.size()) for p in model_parameters])
    return n_params.item()


def json_transform(x):
    if isinstance(x, (int, float, list)): # 默认list中都元素都是数字
        return x
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, dict):
        for k, v in x.items():
            x[k] = json_transform(v)
    return x

def get_monitor_file(task: DictConfig) -> Path:
    
    model = task.model.model
    dataset = f"{task.dataset.dataset}_{task.dataset.name}"
    base_path = Path("monitor_data") / model
    base_path.mkdir(parents=True, exist_ok=True)
    file = base_path / f"{dataset}.json"   
    return file
    
    
if __name__ == "__main__":
    pass
    # m = nn.Linear(10, 20)
    # n = get_n_model_params(m)
    # json.dump({'a': n}, open('test.json', 'w'))