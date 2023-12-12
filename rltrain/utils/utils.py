import os
import torch
import yaml
import collections
from statistics import mean as dq_mean
from typing import Dict, Union, Optional

from rltrain.logger.logger import Logger

# Init CUDA ##############################################################

def init_cuda(gpu: int, cpumin: int, cpumax: int) -> None:

    # BEFORE IMPORTING PYTORCH
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu) # 3 GPU
    os.system("taskset -p -c "+str(cpumin)+"-"+str(cpumax)+" %d" % os.getpid()) #0-1-2 CPU

    # For defining the GPUs: 'nvidia-msi'
    # For defining the CPUs: 'top' and then press '1'

def print_torch_info(logger: Logger, display_mode: bool = False) -> None:
    logger.print_logfile(torch.__version__, display_mode = display_mode)
    logger.print_logfile(str(torch.cuda.is_available()), display_mode = display_mode)
    if torch.cuda.is_available():
        logger.print_logfile(str(torch.cuda.current_device()), display_mode = display_mode)
        logger.print_logfile(str(torch.cuda.device(0)), display_mode = display_mode)
        logger.print_logfile(str(torch.cuda.device_count()), display_mode = display_mode)
        logger.print_logfile(torch.cuda.get_device_name(0), display_mode = display_mode)
    logger.print_logfile("Torch threads: " + str(torch.get_num_threads()), display_mode = display_mode)

# SAVE LOAD YAML ##############################################################

def save_yaml(path: str, data: Dict) -> None:
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)

def load_yaml(file: str) -> Dict:
    if file is not None:
        with open(file) as f:
            return yaml.load(f, Loader=yaml.UnsafeLoader)
    return {}

# DEQUEUE #########################################################################

def safe_dq_mean(dq: collections.deque) -> float:
    return 0.0 if len(dq) == 0 else dq_mean(dq)
