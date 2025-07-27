import numpy as np
import random
import torch
from pathlib import Path
import json 

import dgl 

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


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
        elif isinstance(x, dgl.DGLGraph):
            return f(x)
        else:
            return x

    return _apply(sample)


# def move_to_cuda(sample):
#     def _move_to_cuda(tensor):
#         return tensor.cuda()

#     return apply_to_sample(_move_to_cuda, sample)

def move_to_device(sample, device):
    def _move_to_device(tensor):
        return tensor.to(device)

    return apply_to_sample(_move_to_device, sample)

def move_to_cpu(sample):
    def _move_to_cpu(tensor):
        return tensor.cpu()

    return apply_to_sample(_move_to_cpu, sample)

def convert_tensor_to_str(tensor):
    if torch.is_tensor(tensor):
        return tensor.item()
    return tensor

def convert_dic_to_str(dic):
    result = ''
    if isinstance(dic, str):
        return dic
    if len(dic) == 0:
        return result
    for key, val in dic.items():
        if not key.startswith("_"):
            val = convert_tensor_to_str(val)
            if isinstance(val, int):
                val_str = f"{val}"
            else:
                val_str = f"{val:.4f}" 
            # val_str = f"{val:.4f}" if type(val)==float else f"{val}" # 除了float, 还可能是np.float32/64
            result = result + ' | ' + key + ': ' + val_str
    return result

def add_prefix_to_dict_key(dic, prefix):
    new_dic = {}
    for key, val in dic.items():
        if not key.startswith("_") and not key.endswith("size"):
            new_dic[f"{prefix}{key}"] = val
    return new_dic


def extract_key(dict_list, key):
    return [d[key] for d in dict_list]
    
def average_dict(dict_list):
    # print(f"average over {len(dict_list)} batches")
    if len(dict_list)==0:
        print("empty dict_list in average!")
        return {}
    
    avg_dict = {}
    valid_keys = [key for key in list(dict_list[0].keys()) if not key.startswith("_")]

    # for suffix in suffix_list:
    #     valid_keys += [key for key in list(dict_list[0].keys()) if key.endswith(suffix)]
    #total = sum([d['sample_size'] for d in dict_list])
    for key in valid_keys:
        #avg_dict[key] = sum([d[key]*d['sample_size']/total for d in dict_list])
        if type(dict_list[0][key])==list:
            avg = sum([np.array(d[key]) for d in dict_list])/len(dict_list)
            avg_dict[key] = list(avg)
        else:
            avg_dict[key] = sum([d[key] for d in dict_list])/len(dict_list)
    return avg_dict

# def save_label(dict_list, save_data_dir, epoch):
#     if '_label' in dict_list[0].keys() and '_predict' in dict_list[0].keys():
#         save_path = save_data_dir / f"label_vs_predict_{epoch}.json"
#         label, predict = [], []
#         for d in dict_list:
#             label += d['_label']
#             predict += d['_predict']
#         save_d = {'label': label, 'predict': predict}
#         json.dump(save_d, open(save_path, 'w'), indent=2)


