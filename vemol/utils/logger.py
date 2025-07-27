import argparse
import os
import warnings

logger = None


class _Logger:
    def __init__(self, filename=None, path='.', only_print=False, append=False):
        os.makedirs(path, exist_ok=True)
        self.filename = os.path.join(path, filename)
        self.file = None
        if filename and not only_print:
            self.file = open(self.filename, 'a' if append else 'w')

    def __del__(self):
        if self.file:
            self.file.close()
            
    def log(self, msg='', end='\n', is_print=True, is_log=True):
        return self.__call__(msg, end, is_print, is_log)

    def __call__(self, msg='', end='\n', is_print=True, is_log=True):
        if is_print:
            print(msg, end=end)
        if is_log and self.file is not None:
            self.file.write(msg)
            self.file.write(end)
            self.file.flush()


def setting_logger(filename=None, path='.', only_print=False, append=False):
    global logger
    logger = _Logger(filename, path, only_print, append)
    return logger


def get_logger():
    global logger
    if logger is None:
        # warnings.warn('Logger is not set!')
        return print
    else:
        return logger


import logging
import sys
import os
import time
 
import hydra 

# def setup_logger(save_dir, distributed_rank=0, filename="output.log"):
#     # if current process is not master process, we create a child logger for it,
#     # and don't propagate this child logger's message to the root logger.
#     # We don't create any handlers to this child logger, so that no message will be ouput from this process.
#     if distributed_rank > 0: 
#         logger_not_root = logging.getLogger(name=__name__) 
#         logger_not_root.propagate = False
#         return logger_not_root
     
#     # if current process is master process, we create a root logger for it,
#     # and create handlers for the root logger.
#     root_logger = logging.getLogger()
#     # root_logger.setLevel(logging.INFO)
#     # ch = logging.StreamHandler(stream=sys.stdout)
#     # ch.setLevel(logging.INFO)
#     formatter = logging.Formatter("%(asctime)s | %(levelname)s: %(message)s")
#     # ch.setFormatter(formatter)
#     # root_logger.addHandler(ch)
#     # print("handlers:", root_logger.handlers)
#     # exit()
#     # save_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
#     filename = 'train.log'
#     if save_dir:
#         save_file = os.path.join(save_dir, filename)
#         if not os.path.exists(save_file):
#             os.system(r"touch {}".format(save_file))
#         fh = logging.FileHandler(save_file, mode='a')
#         fh.setLevel(logging.INFO)
#         fh.setFormatter(formatter)
#         root_logger.addHandler(fh)
     
#     return root_logger


def setup_logger(save_dir, distributed_rank=0, filename="output.log"):
    # if current process is not master process, we create a child logger for it,
    # and don't propagate this child logger's message to the root logger.
    # We don't create any handlers to this child logger, so that no message will be ouput from this process.
    if distributed_rank > 0: 
        logger_not_root = logging.getLogger(name=__name__) 
        logger_not_root.propagate = False
        return logger_not_root
     
    # if current process is master process, we create a root logger for it,
    # and create handlers for the root logger.
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    root_logger.addHandler(ch)
 
    if save_dir:
        save_file = os.path.join(save_dir, filename)
        if not os.path.exists(save_file):
            os.system(r"touch {}".format(save_file))
        fh = logging.FileHandler(save_file, mode='a')
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        root_logger.addHandler(fh)
     
    return root_logger