from pathlib import Path 

# from typing_extensions import Literal
# from typing import Literal 

DATASET_CACHE_DIR = Path("data")

SPLIT_TO_ID = {'train':0, 'valid':1, 'test':2}

BENCHMARK = {
             'polymer_pp':['polymer_eat', 'polymer_eea', 'polymer_egb', 'polymer_egc', 
                            'polymer_ei', 'polymer_eps', 'polymer_nc', 'polymer_xc', 'pl1m',
                            'polymer_eat_aug', 'polymer_eea_aug', 'polymer_egb_aug', 'polymer_egc_aug', 
                            'polymer_ei_aug', 'polymer_eps_aug', 'polymer_nc_aug', 'polymer_xc_aug', 'pl1m_aug']} 

CLS_TASKS = {}


CLS_METRICS = ['rocauc', 'ap', 'acc', 'f1'] # 这里ap就是aupr值
MULTI_CLS_METRICS = ['acc'] 
REG_METRICS = ['rmse', 'mae', 'r2']
METRIC_BEST_TYPE = {'rocauc': 'max', 'ap': 'max', 'acc': 'max', 'f1': 'max', 'rmse': 'min', 'mae': 'min', 'r2': 'max'}
METRICS = {'cls': CLS_METRICS, 'reg': REG_METRICS, 'multi_cls': MULTI_CLS_METRICS}


def get_task_type(dataset): # -> Literal['cls', 'reg', 'multi_cls']
    if dataset in CLS_TASKS:
        return 'cls'
    else:
        return 'reg'

def get_task_metrics(dataset):
    return METRICS[get_task_type(dataset)]

def get_split_name(dataset, scaffold_id):
    return f"kfold-{scaffold_id}"
    
DATASET_TASKS = {}

# 其余数据集的任务数均为1
for name, ds in BENCHMARK.items():
    for d in ds:
        if d not in DATASET_TASKS:
            DATASET_TASKS[d] = 1

BENCHMARK_NAME = {} # 数据集名字映射到所在的benchmark的名字
for name, ds in BENCHMARK.items():
    for d in ds:
        BENCHMARK_NAME[d] = name 

BENCHMARK_BASE_PATH = {
             'polymer_pp': Path('data/polymer_pp')
             }


DATASET_SIZE = {
    'pl1m': 995797,
    'pl1m_aug': 995797,
}