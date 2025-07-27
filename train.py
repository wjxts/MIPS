import logging
import os
import sys

import torch
import torch.multiprocessing as mp
import torch.distributed as dist

import hydra
import numpy as np

from omegaconf import DictConfig, OmegaConf

from vemol.trainer import Trainer
from vemol.dataclass.initialize import add_defaults
from vemol.dataclass.config import post_init

# logger = logging.getLogger(__name__)

# to make the tuple work in hydra
def resolve_tuple(*args):
    return tuple(args)

OmegaConf.register_new_resolver('as_tuple', resolve_tuple)

def get_experiment_name(cfg):
    experiment_name = f"{cfg.model.model}_{cfg.dataset.dataset}"
    experiment_name += f"_{cfg.optimizer.optimizer}"
    experiment_name += f"_{cfg.criterion.criterion}"
    experiment_name += f"_{cfg.scheduler.scheduler}"

    print(f"Experiment:{experiment_name}")
    return experiment_name


@hydra.main(version_base=None, config_path=os.path.join("vemol", "config"), config_name="mpp")
def main(cfg: DictConfig):
    # add_defaults(cfg)
    # cmd_line_args = cmd_line_args = sys.argv[1:]
    cfg = OmegaConf.create(cfg) # can be removed
    # print("here")
    log_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    cfg.common.log_dir = log_dir
    
    # allow custom configuration in task python file
    post_init(cfg, cfg.local_config) 
    
    # override the configuration with cli arguments
    override_file = os.path.join(log_dir, ".hydra", "overrides.yaml") 
    override_config_list = OmegaConf.load(override_file)
    override_config_list = list(override_config_list)
    override_config_list = [l for l in override_config_list if "." in l] 
    # 这里如果是override node，则不能在这里merge; 只有value需要merge; 这里假设了只有cfg只有一层, 如果是多层也不能这么处理
    override_cfg = OmegaConf.from_dotlist(override_config_list)
    if 'model' in override_cfg and ('input_dim' in override_cfg.model or 'output_dim' in override_cfg.model):
        raise ValueError("input_dim and output_dim should not be overrided in model, should be set in dataset")
        
    cfg = OmegaConf.merge(cfg, override_cfg) 
    
    # save the updated config
    yaml_file_path = os.path.join(log_dir, ".hydra", "updated_config.yaml")  # save the updated config
    OmegaConf.save(config=cfg, f=yaml_file_path)
    
    # print(logger.handlers) # 虽然为[], 但是后边的logger都可以正确记录, 同时在console和文件中存储logging
    # logger.info(f"task: {cfg.task}")
    torch.set_num_threads(cfg.common.num_threads)
    # get_name
    # if len(cmd_line_args) > 0 and (not cfg.checkpoint.resume):
    #     cfg.name = cfg.name + '_'.join(cmd_line_args).replace('=', '').replace('common.', '')
    cfg.common.data_parallel_size = torch.cuda.device_count()
    cfg.model.data_parallel_size = cfg.common.data_parallel_size
    cfg.dataset.data_parallel_size = cfg.common.data_parallel_size
    
    if cfg.common.device == 0:
        print(f"Log dir: {log_dir}")
        print(f"data_parallel_size: {cfg.common.data_parallel_size}")
    if cfg.common.data_parallel_size > 1:
        mp.spawn(_train, args=(cfg, ), nprocs=cfg.common.data_parallel_size)
    else:
        _train(cfg=cfg)

def _train(rank=0, cfg=None):
    cfg.common.device = rank
    cfg.dataset.device = cfg.common.device
    cfg.model.device = cfg.common.device
    
    if cfg.common.data_parallel_size > 1:
        ddp_setup(rank=rank, world_size=cfg.common.data_parallel_size)
    
    train = Trainer(cfg)
    train.train()   
    
    if cfg.common.data_parallel_size > 1:
        dist.destroy_process_group()  
    
    
def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    
# def main_cli():
#     main()

if __name__ == "__main__":
    main()
