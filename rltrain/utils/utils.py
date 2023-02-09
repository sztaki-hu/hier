import os
import torch

# Init CUDA ##############################################################

def init_cuda(gpu,cpumin,cpumax):

    # BEFORE IMPORTING PYTORCH
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu) # 3 GPU
    os.system("taskset -p -c "+str(cpumin)+"-"+str(cpumax)+" %d" % os.getpid()) #0-1-2 CPU

    # For defining the GPUs: 'nvidia-msi'
    # For defining the CPUs: 'top' and then press '1'

def print_torch_info(logger):
    logger.print_logfile(torch.__version__)
    logger.print_logfile(torch.cuda.is_available())
    if torch.cuda.is_available():
        logger.print_logfile(torch.cuda.current_device())
        logger.print_logfile(torch.cuda.device(0))
        logger.print_logfile(torch.cuda.device_count())
        logger.print_logfile(torch.cuda.get_device_name(0))
    logger.print_logfile("Torch threads: " + str(torch.get_num_threads()))
