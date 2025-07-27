import os
import argparse
from collections import OrderedDict

from dataclasses import dataclass, field
import torch
from hydra.core.config_store import ConfigStore
from omegaconf import II, MISSING

from vemol.dataclass import BaseDataclass
# from vemol.utils.logger import get_logger
import logging

from pathlib import Path 

@dataclass
class CheckpointConfig(BaseDataclass):
    save_dir: str = field(
        default='./checkpoint/' + II('task') + '/' + II('name'), metadata={'help': "root path to the checkpoint needed resume and load."}
    )
    resume: bool = False
    resume_file: str = field(
        default='latest_checkpoint.pth', metadata={'help': "path to the checkpoint needed resume."}
    )
    # may need an argument to decide relative model path or not. Not needed for now.
    load_model: str = field(
        default="", metadata={'help': "The path to (pre-)trained model."}
    )
    evaluate_model_path: str = field(
        default="average_checkpoint.pth",
        metadata={'help': "(relative) model path for evaluation"}
    )
    # may need an argument to decide relative model path or not. Not needed for now.
    load_no_strict: bool = field(
        default=False, metadata={'help': "The keys of loaded model may not exactly match the model\'s. (May usefully for finetune)."}
    )
    save_to_disk: bool = field(
        default=True, metadata={'help': "whether save to disk"}
    )
    save_epoch: bool = field(
        default=True, metadata={'help': "whether save according to epochs"}
    )
    save_step: bool = field(
        default=False, metadata={'help': "whether save according to steps"}
    )
    save_epoch_interval: int = field(
        default=1, metadata={"help": "save a checkpoint every N epochs"}
    )
    save_step_interval: int = field(
        default=1000, metadata={"help": "save a checkpoint every M steps"}
    )
    data_parallel_size: int = II("common.data_parallel_size")
    keep_last_epochs: int = 5
    

cs = ConfigStore.instance()
name = 'base_checkpoint'
node = CheckpointConfig()
node._name = name
cs.store(name=name, group="checkpoint", node=node)

def _strip_prefix_if_present(state_dict, prefix):
    # 注意ddp wrap的model state_dict的key前边多了"module."前缀, model里所有模块命名不要带有module.字段，否则会出错
    keys = sorted(state_dict.keys())
    if not all(key.startswith(prefix) for key in keys):
        return state_dict
    stripped_state_dict = OrderedDict()
    for key, value in state_dict.items():
        stripped_state_dict[key.replace(prefix, "")] = value
    return stripped_state_dict


class Checkpoint(object):
    checkpoint = None

    def __init__(self, task, model, optimizer=None, scheduler=None, logger=None):
        self.model = model
        self.task = task
        self.cfg = task.checkpoint
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_dir = self.cfg.save_dir
        self.save_to_disk = self.cfg.save_to_disk and bool(self.cfg.save_dir)
        if logger is None:
            # logger = get_logger(__name__)
            logger = logging.getLogger(__name__)
        self.logger = logger
        if not os.path.exists(self.cfg.save_dir):
            os.makedirs(self.cfg.save_dir, exist_ok=True)
        self.keep_last_epochs = self.cfg.keep_last_epochs
        
    @property
    def evaluate_model_path(self):
        return os.path.join(self.save_dir, self.cfg.evaluate_model_path)
    
    @property
    def resume_model_path(self):
        return os.path.join(self.save_dir, self.cfg.resume_file)
    
    @property
    def load_model_path(self):
        return self.cfg.load_model
    
    def _check_name(self, name: str):
        if not name.endswith('.pth'):
            name = name + '.pth'
        return os.path.join(self.cfg.save_dir, name)

    def save_checkpoint(self, name, epoch=-1, **kwargs):
        '''
        name can be epoch or "best"
        '''
        is_best = (name == "best")
        if not self.cfg.save_to_disk:
            return
        ckpt_name = 'checkpoint_' + str(name)
        save_file = self._check_name(ckpt_name)
        data = {}
        # DataParallel or DistributedDataParallel during serialization,
        # remove the "module" prefix before saving
        data["model"] = _strip_prefix_if_present(self.model.state_dict(), prefix="module.")
        data["task"] = self.task
        data["optimizer"] = self.optimizer.state_dict() if self.optimizer is not None else None
        data["scheduler"] = self.scheduler.state_dict() if self.scheduler is not None else None
        data["epoch"] = epoch
        data.update(kwargs)
        if self.logger is not None:
            self.logger.info("Saving checkpoint to {}".format(save_file))

        torch.save(data, save_file)  # self.tag_last_checkpoint(save_file)
        if not is_best:
            torch.save(data, os.path.join(self.cfg.save_dir, 'latest_checkpoint.pth'))

        if self.keep_last_epochs>0 and type(name)==int:
            epoch = name 
            if epoch <= self.keep_last_epochs:
                return
            remove_file = Path(self._check_name(f'checkpoint_{epoch-self.keep_last_epochs}'))
            self.logger.info(f"remove {remove_file}")
            if remove_file.exists(): # 可能存在并发问题
                remove_file.unlink() # 
            
    def save_model(self, name=0):      
        # deprecated, should not be used
        if not self.save_to_disk:
            return
        name = 'model_' + str(name)
        save_file = self._check_name(str(name))
        data = _strip_prefix_if_present(self.model.state_dict(), prefix="module.")
        self.logger.info("Saving model to {}".format(save_file))
        torch.save(data, save_file)

    def load(self, f=None):
        # if self.has_checkpoint():
        # override argument with existing checkpoint
        # f = self.get_checkpoint_file()
        if not f:
            # no checkpoint could be found
            raise ValueError("No checkpoint found.")
        if self.logger is not None:
            self.logger.info(f"==> Loading model from {f}, strict: {not self.cfg.load_no_strict}")
        checkpoint = torch.load(f, map_location=torch.device("cpu"))
        # if the state_dict comes from a model that was wrapped in a
        # DataParallel or DistributedDataParallel during serialization,
        # remove the "module" prefix before performing the matching
        # checkpoint['model'] = self._strip_prefix_if_present(checkpoint['model'], prefix="module.")
        # print(list(self.model.state_dict().keys()))[:2]
        # print(list(checkpoint['model'].keys()))[:2];exit()
        self.model.load_state_dict(checkpoint['model'], strict=(not self.cfg.load_no_strict))
        self.logger.info("load model successfully")
        return checkpoint

    def resume(self, f=None):
        # if self.has_checkpoint():
        # override argument with existing checkpoint
        # f = self.get_checkpoint_file()
        if not f:
            # no checkpoint could be found
            raise ValueError("No resume checkpoint found! Please assign one.")
        # f = os.path.join(self.cfg.save_dir, f) # use full path!
        if self.logger is not None:
            self.logger.info("Loading checkpoint from {}".format(f))
        # if Checkpoint.checkpoint is not None:
        #     checkpoint = Checkpoint.checkpoint
        #     Checkpoint.checkpoint = None
        # else:
        # print("f:", f);exit()
        checkpoint = torch.load(f, map_location=torch.device("cpu"))
        #import pdb
        #pdb.set_trace()
        # if the state_dict comes from a model that was wrapped in a
        # DataParallel or DistributedDataParallel during serialization,
        # remove the "module" prefix before performing the matching
        
        # checkpoint["model"] = _strip_prefix_if_present(checkpoint["model"], prefix="module.")
        self.model.load_state_dict(checkpoint["model"], strict=(not self.cfg.load_no_strict))
        
        if "optimizer" in checkpoint and self.optimizer:
            if self.logger is not None:
                self.logger.info("Loading optimizer from {}".format(f))
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint and self.scheduler:
            if self.logger is not None:
                self.logger.info("Loading scheduler from {}".format(f))
            self.scheduler.load_state_dict(checkpoint["scheduler"])
        if "cfg" in checkpoint:
            self.cfg = checkpoint["cfg"]
        self.logger.info("resume model successfully")
        start_epoch = checkpoint.get("epoch", 0) + 1
        #self.logger.info(self.cfg)
        # return any further checkpoint data
        return start_epoch

    def has_checkpoint(self):
        save_file = os.path.join(self.cfg.save_dir, "last_checkpoint")
        return os.path.exists(save_file)

    def get_checkpoint_file(self):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        try:
            with open(save_file, "r") as f:
                last_saved = f.read()
        except IOError:
            # if file doesn't exist, maybe because it has just been
            # deleted by a separate process
            last_saved = ""
        return last_saved

    def tag_last_checkpoint(self, last_filename):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        with open(save_file, "w") as f:
            f.write(last_filename)

    @staticmethod
    def load_config(f=None):
        if f:
            Checkpoint.checkpoint = torch.load(f, map_location=torch.device("cpu"))
            if "cfg" in Checkpoint.checkpoint:
                print('Read config from checkpoint {}'.format(f))
                return Checkpoint.checkpoint.pop("cfg")
        return None
