from os.path import dirname, abspath

import numpy as np
import torch
import argparse

from multiprocessing import Process
from rltrain.buffers.demo import Demo

from rltrain.utils.utils import init_cuda, print_torch_info
from rltrain.logger.logger import Logger

import datetime
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

import torch
    

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--configfile", default="cfg/config.yaml" ,help="Path of the config file")
    #parser.add_argument("--configfile", default="logs/0216_A_test_stack_blocks_sac" ,help="Path of the config file")
    parser.add_argument("--trainid", type=int, default=0 ,help="Train ID")
    parser.add_argument("--restart", type=bool, default=False ,help="Set true if you want to restart a training")
    parser.add_argument("--restart_epoch", type=int, default=5 ,help="The epoch number from where you want to restart the training.")
    # Example: python3 main.py --configfile /cfg/alma.yaml
    args = parser.parse_args()

    # Init logger ###############################################x
    current_dir = dirname(abspath(__file__))
    logger = Logger(current_dir = current_dir, main_args = args, light_mode = True)
    config = logger.get_config()

    #print(config['environment']['task']['params'])
    
    # Init CUDA and torch and np ##################################
    init_cuda(config['hardware']['gpu'][args.trainid],config['hardware']['cpu_min'][args.trainid],config['hardware']['cpu_max'][args.trainid])

    print_torch_info(logger)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    torch.set_num_threads(torch.get_num_threads())

    torch.manual_seed(config['general']['seed'])
    np.random.seed(config['general']['seed'])

    # Init Demo Buffer ################################################
    
    demo = Demo(logger,config)

    demo.clean_up_old_demo()
    demo.create_demos()

    # demoBuffer = demo.load_demos()
    # batch = demoBuffer.get_first(10)
    # print(batch)
                     

if __name__ == '__main__':
    main()

    
