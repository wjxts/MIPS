# standard library
import logging
import json 
from pathlib import Path 
from tqdm import tqdm

# third-party library
# import numpy as np
from omegaconf import DictConfig
import torch
# from torch import autograd
# import torch.nn as nn
# import torch.distributed as dist

logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR) # for wandb network warning
import wandb

# local modules
from vemol.criterion import build_criterion
from vemol.metric import build_metric 
from vemol.dataset import build_dataloader
from vemol.dataset.iterators import GroupedIterator
# from vemol.generator import build_generator
from vemol.models import build_model
from vemol.models.ema_model import build_ema_model
from vemol.optimizer import build_optimizer
# from vemol.resource import build_common_resource
from vemol.scheduler import build_scheduler
from vemol.utils.checkpoint import Checkpoint
from vemol.utils.distributed import mean_reduce
from vemol.utils.logger import setup_logger
from vemol.utils.process_data import add_prefix_to_dict_key, average_dict, convert_dic_to_str
from vemol.utils.process_data import move_to_device, set_random_seed
from vemol.utils.utils import is_better, empty_context, get_result_file_from_cfg, json_transform
from vemol.utils.utils import collect_test_results, cal_grad_norm, get_n_model_params
from vemol.utils.timer import Timer

# local third-party library





class Trainer(object):
    def __init__(self, task: DictConfig):
        # self.date = datetime.datetime.now()
        # print(task);exit()
                
        self.task = task
        
        if self.task.common.save_results:
            self.save_result_file = get_result_file_from_cfg(self.task)
            if self.save_result_file.exists():
                print(f"File {self.save_result_file} exists")
                exit()
        
        if self.task.common.device == 0:
            if task.common.data_parallel_size > 1:
                # self.logger = setup_logger(task.checkpoint.save_dir, 0)
                self.logger = setup_logger(task.common.log_dir)
            else:
                # self.logger = logging.getLogger(task.task)
                self.logger = logging.getLogger(__name__) # __name__ = vemol.trainer, 似乎是这个文件在package的相对路径
                
        set_random_seed(task.common.seed)
        # 0. build common resources and generator (global objects)
        # self.common_resource = build_common_resource(task)
        # self.generator = build_generator(task)
        
        # init dataloader task define data loading
        self.dataloader = build_dataloader(task.dataset)
        
        # align the model dimension with dataset dimension
        task.model.input_dim = task.dataset.input_dim
        task.model.output_dim = task.dataset.output_dim
        
        # init model  task define model transform
        self.model = build_model(task.model)
        self.ema_model = build_ema_model(self.model, task.common.ema_decay)
        # init criterion task define criterion
        self.criterion = build_criterion(task.criterion)
        self.criterion.set_model(self.model)
        # init validation metric if validate = True
        if task.common.validate:
            self.metric = build_metric(task.metric)
        
        # init optimizer
        self.optimizer = build_optimizer(task.optimizer)
        self.optimizer.set_model(self.model)
        # init learning rate scheduler
        task.scheduler.num_batches_per_epoch = self.dataloader.num_batches_per_epoch
        self.scheduler = build_scheduler(task.scheduler, self.optimizer.get_optimizer())
        if self.task.common.device == 0:
            self.logger.info(f"init learning rate: {self.scheduler.get_lr()}")
        # init logger and checkpoint
        
        
        
        # 似乎用self.model.get_model()更好
        self.checkpoint = Checkpoint(task=task, model=self.model, optimizer=self.optimizer.get_optimizer(), scheduler=self.scheduler)
        
            
        # self.scaler = torch.cuda.amp.GradScaler() # for mixed precision training
        self.scaler = torch.amp.GradScaler('cuda') # for mixed precision training
        # self.scaler = torch.cuda.amp.GradScaler(init_scale=2.**1, growth_interval=10)
        
        # prepare wandb and monitor
        self.use_wandb = self.task.common.wandb and self.task.common.device == 0
        self.use_monitor = self.task.common.monitor and self.task.common.device == 0
        # if self.use_monitor:
        #     assert self.use_wandb==True, "monitor must be used with wandb" # 不是必须的, 可以单独使用monitor
        # print(self.task.monitor);exit()
        if self.use_monitor:
            from Taiyi.taiyi.monitor import Monitor
            self.monitor = Monitor(self.model, self.task.monitor_config)
        else:
            self.monitor = None
        
        if self.task.common.device == 0:
            visualize = wandb if self.use_wandb else None
            from Taiyi.visualize import Visualization # 可以单独把这个文件放在utils下边
            self.visualize = Visualization(task, self.monitor, visualize)
            
        self.finish = False
        
        
        # self.ext = {} # external data to display
        
    def train(self):
        # best_valid_index = -1e8 # wrong! some indices are the smaller the better
        
        self.timer = Timer()
        best_valid_index = None 
        start_epoch = self.task.common.start
        if self.task.checkpoint.load_model != "":
            self.checkpoint.load(self.checkpoint.load_model_path) # load_model_path=checkpoint.load_model
        if self.task.checkpoint.resume:
            start_epoch = self.checkpoint.resume(self.checkpoint.resume_model_path)
        
        if self.task.common.device == 0:
            self.checkpoint.save_checkpoint("init")
        # if not self.task.checkpoint.resume:
        #     # save model before training
        #     self.checkpoint.save_checkpoint(name=0, epoch=0)
            
        if self.task.common.data_parallel_size > 1:
            # wrap ddp after load state_dict !!!
            self.model.wrap_ddp()
        
        if self.task.common.max_steps > 0: # 优先按照max_steps设置
            self.task.common.epochs = self.task.common.max_steps // len(self.dataloader.get_dataloader('train')) + 1
        else:
            self.task.common.max_steps = self.task.common.epochs * len(self.dataloader.get_dataloader('train'))
        
        
        train_score_list = []
        valid_score_list = []
        test_score_list = []
        if self.ema_model is not None:
            ema_valid_score_list = []
            ema_test_score_list = []
        for epoch in range(start_epoch-1, self.task.common.epochs + 1):
            # the first epoch is epoch 0, only for validate
            if self.finish:
                break 
            self.task.common.global_epoch = epoch
            
            # Training one epoch
            if epoch>=start_epoch:
                if self.task.common.device == 0:
                    self.logger.info(f"epoch:{epoch} learning rate: {self.scheduler.get_lr()}")
                train_result = self.train_epoch(epoch)
            
                if self.task.common.data_parallel_size > 1:
                    mean_reduce(train_result, self.task.common.data_parallel_size, 0)
            
                if self.task.common.device == 0:
                    self.logger.info(f"epoch:{epoch} train" + convert_dic_to_str(train_result))
                    
                    train_result = add_prefix_to_dict_key(train_result, "train_")
                    self.visualize.log_ext(train_result, name=f"train_epoch_{epoch}")
            
            # Validate the model
            if epoch % self.task.common.val_interval == 0 and self.task.common.validate:
                val_result = self.validate(self.model, epoch, split='valid')
                test_result = self.validate(self.model, epoch, split='test')
                if self.ema_model is not None:
                    ema_val_result = self.validate(self.ema_model, epoch, split='valid')
                    ema_test_result = self.validate(self.ema_model, epoch, split='test')
                if self.task.common.data_parallel_size > 1:
                    mean_reduce(val_result, self.task.common.data_parallel_size, 0)
                    mean_reduce(test_result, self.task.common.data_parallel_size, 0)
                    if self.ema_model is not None:
                        mean_reduce(ema_val_result, self.task.common.data_parallel_size, 0)
                        mean_reduce(ema_test_result, self.task.common.data_parallel_size, 0)
                
                current_valid_index = val_result[self.task.criterion.valid_criterion]
                
                # take care of "<" comparison! the index should be the larger, the better
                
                save_checkpoint = (best_valid_index is None) or is_better(current_valid_index, best_valid_index, self.task.criterion.valid_criterion)
                if self.task.common.device == 0 and save_checkpoint:
                    best_valid_index = current_valid_index
                    self.checkpoint.save_checkpoint('best', epoch=epoch)
            
                if self.task.common.device == 0:
                    self.logger.info(f"epoch:{epoch} validation" + convert_dic_to_str(val_result))
                    self.logger.info(f"epoch:{epoch} test" + convert_dic_to_str(test_result))
                    log_val_result = add_prefix_to_dict_key(val_result, "valid_")
                    log_test_result = add_prefix_to_dict_key(val_result, "test_")
                    self.visualize.log_ext(log_val_result, name=f"valid_epoch_{epoch}")
                    self.visualize.log_ext(log_test_result, name=f"test_epoch_{epoch}")
                    valid_score_list.append(val_result)
                    test_score_list.append(test_result)
                    if self.ema_model is not None:
                        self.logger.info(f"epoch:{epoch} ema_validation" + convert_dic_to_str(ema_val_result))
                        self.logger.info(f"epoch:{epoch} ema_test" + convert_dic_to_str(ema_test_result))
                        log_ema_val_result = add_prefix_to_dict_key(ema_val_result, "ema_valid_")
                        log_ema_test_result = add_prefix_to_dict_key(ema_val_result, "ema_test_")
                        self.visualize.log_ext(log_ema_val_result, name=f"ema_valid_epoch_{epoch}")
                        self.visualize.log_ext(log_ema_test_result, name=f"ema_test_epoch_{epoch}")
                        ema_valid_score_list.append(ema_val_result)
                        ema_test_score_list.append(ema_test_result)
                
            self.task.common.start += 1

            # if self.vis and self.task.common.device == 0:
            #     self.visualize.log_ext(self.ext)
            
            if self.task.checkpoint.save_epoch and epoch % self.task.checkpoint.save_epoch_interval == 0 and self.task.common.device == 0:
                self.checkpoint.save_checkpoint(epoch, epoch=epoch)

            if self.ema_model is not None:
                self.ema_model.update_ema(self.model)
        # if self.use_wandb:
        #     wandb.finish()
        
        if self.task.common.device == 0:
            self.visualize.close()
        
        # collect test results
        if self.task.common.device == 0 and self.task.common.save_results and self.task.common.validate:
            if self.ema_model is not None:
                d_result = collect_test_results(train_score_list, valid_score_list, test_score_list, 
                                                ema_valid_score_list, ema_test_score_list)
            else:
                d_result = collect_test_results(train_score_list, valid_score_list, test_score_list)
            
        elapsed_time = self.timer.elapsed()
        if self.task.common.device == 0:
            self.logger.info(f"Training finished! Total time: {elapsed_time}")
        
        if self.task.common.device == 0 and self.task.common.save_results:
            d_result['folder'] = self.task.common.log_dir
            d_result['n_params'] = get_n_model_params(self.model)
            json.dump(d_result, open(self.save_result_file, 'w'), indent=2)
            self.logger.info(f"Save results to {self.save_result_file}")
        
        if self.task.common.dump_monitor and self.task.common.device == 0:
            self.dump_monitor()
            
    def dump_monitor(self):
        from vemol.utils.utils import get_monitor_file
        monitor_output = self.monitor.get_output()
        monitor_output = json_transform(monitor_output)
        # print(monitor_output);exit()
        save_file = get_monitor_file(self.task)
        json.dump(monitor_output, open(save_file, 'w'), indent=2)
            
    def train_epoch(self, epoch):
        self.model.train()
        self.criterion.train()
        # self.criterion.start(mode='train')
        dataloader = self.dataloader.get_dataloader('train', epoch)
        dataloader = GroupedIterator(dataloader, self.task.common.update_freq)
        # 可以用一个base类包装一下，这样就不用import GroupedIterator类了
        # lr.step
        result_logs = []
        for step, samples in enumerate(dataloader):
            if self.finish:
                break 
            
            # 用于记录训练时间的比例, 来观察dataloader的效率
            self.timer.start_train_step()
            # train one step
            # with autograd.detect_anomaly():
            result_log = self._train_step(samples)
            result_logs.append(result_log)
            
            # 用于记录训练时间的比例, 来观察dataloader的效率
            self.timer.end_train_step()
            
            self.timer.step()
            if step % self.task.common.log_interval == 0 and self.task.common.device == 0:
                # self.logger.info(f"epoch:{epoch} | step:{step} | global_step:{self.task.common.global_step}/{self.task.common.max_steps} | GPU:{self.task.common.device}" + convert_dic_to_str(result_log))
                progress_ratio = self.task.common.global_step / self.task.common.max_steps
                progress_str = f"global_steps: {self.task.common.global_step}/{self.task.common.max_steps}({progress_ratio*100:.2f}%)"
                elapsed_time = self.timer.elapsed()
                
                eta = self.timer.eta(self.task.common.max_steps-self.task.common.global_step)
                time_str = f"ET:{elapsed_time}(ETA:{eta})"
                
                train_time_ratio = self.timer.train_time_ratio
                
                self.logger.info(f"epoch:{epoch} | step:{step} | {progress_str} | {time_str} | {train_time_ratio}" + convert_dic_to_str(result_log))
                
                # print(f"scaler: {self.scaler._scale}")

        # prepare the learning rate for the next epoch
        self.scheduler.epoch_update(epoch+1) # 还有一个step_update
        
        # save_json(extract_key(result_logs, "loss"), "train_result.json");exit()
        if self.task.common.device == 0:
            self.visualize.periodic_save_to_local(name=f"step_data_epoch_{epoch}")
        result_log_avg = average_dict(result_logs)
        return result_log_avg
        # return self.criterion.end(result_log)
        
    def _train_step(self, samples):
        
        samples = move_to_device(samples, self.task.common.device) # 多卡似乎需要在这里将数据搬到gpu
        # zero grad
        self.optimizer.zero_grad()
        
        for i, sample in enumerate(samples):
            def maybe_no_sync():
                """
                Whenever *samples* contains more than one mini-batch, we
                want to accumulate gradients locally and only call
                all-reduce in the last backwards pass.
                """
                if (
                    self.task.common.data_parallel_size > 1
                    and i < len(samples) - 1
                ):
                    return self.model.get_model().no_sync()
                else:
                    return empty_context()
            
            with maybe_no_sync():
                # with torch.cuda.amp.autocast(enabled=self.task.common.mixed_precision):
                with torch.amp.autocast('cuda', enabled=self.task.common.mixed_precision):
                    output = self.model.train_step(sample) 
                    # cal loss
                    labels = sample['labels']
                    result = self.criterion(output, labels)
                    loss = result['loss']/len(samples)
                    
                    # backward
                    if self.task.common.mixed_precision:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()
        
        if self.task.optimizer.clip_norm > 0:
            if self.task.common.mixed_precision:
                self.scaler.unscale_(self.optimizer.get_optimizer())
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.task.optimizer.clip_norm)
        
        # print grad norm 
        # grad_norm = cal_grad_norm(self.model)
        # self.logger.info(f"grad_norm: {grad_norm}");exit()
        # monitor before optimizer update parameters
        if self.use_monitor:
            self.monitor.track(self.task.common.global_step) 
        if self.task.common.device == 0:
            self.visualize.show(self.task.common.global_step, 
                                ext=add_prefix_to_dict_key(result['log'], "train_step_"))
        
        # optimizer.step
        if self.task.common.mixed_precision:
            self.scaler.step(self.optimizer.get_optimizer())
            self.scaler.update()
        else:
            self.optimizer.step()
            
        # lr_scheduler.step() after optimizer.step(), for pytorch Scheduler Class
        
        
        if self.task.checkpoint.save_step and self.task.common.global_step % self.task.checkpoint.save_step_interval == 0 and self.task.common.device == 0:
            self.checkpoint.save_checkpoint(f"step_{self.task.common.global_step}")
        
        if self.task.common.global_step >= self.task.common.max_steps:
            self.finish = True
        self.task.common.global_step += 1
        
        self.scheduler.step_update(self.task.common.global_step) # after optimizer.step() and step += 1
        
        return result['log']
    
    def _validate_step(self, model, sample):
        # sample = move_to_device(sample, self.task.common.device) # 放到validate里边
        # if self.task.common.data_parallel_size > 1:
        #     output = self.model.module.valid_step(sample)
        # else:
        #     output = self.model.valid_step(sample)
        predicts = model.valid_step(sample)
        return predicts
    
    def validate(self, model, epoch, split='valid'):
        #self.criterion.start('val')
        # 多卡这里会有问题, 两半样本的平均auc不等于全部样本的auc
        self.model.eval()
        self.criterion.eval()
        predict_list = []
        label_list = []
        with torch.no_grad():
            for sample in tqdm(self.dataloader.get_dataloader(split)):
                sample = move_to_device(sample, self.task.common.device)
                predicts = self._validate_step(model, sample) # 数据集应该没大到没次只能算一个batch的metric, 应该可以先把结果保存, 最后再计算metric
                
                labels = sample['labels']
                predict_list.append(predicts)
                label_list.append(labels)
        
        # calculate metric
        from vemol.utils.utils import merge_list
        predicts = merge_list(predict_list)
        labels = merge_list(label_list)
        # predicts = torch.cat(predict_list, dim=0) 
        # labels = torch.cat(label_list, dim=0)
        result_log = self.metric(predicts, labels)
        # print(result_log);exit()
        # print(result_logs)
        # if self.task.common.device == 0:
            # only support single GPU test
            #save_label(result_logs, self.save_data_dir, epoch)
            # self.visualize.save_label_and_predict(result_log, epoch)
        return result_log
        #return self.criterion.end()
        
    
        
        