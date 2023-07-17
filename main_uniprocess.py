from os.path import dirname, abspath

import numpy as np
import torch
import argparse

from rltrain.agents.agent_v0 import Agent

from rltrain.buffers.replay_v0 import ReplayBuffer

from rltrain.utils.utils import init_cuda, print_torch_info
from rltrain.logger.logger import Logger
from rltrain.envs.builder import make_env
from rltrain.runners.uniprocess.sampler_trainer_tester_v2 import SamplerTrainerTester


import datetime
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

import torch
import pandas as pd
    

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--configfile", default="cfg/config_uni_spinup.yaml" ,help="Path of the config file")
    #parser.add_argument("--configfile", default="logs/0216_A_test_stack_blocks_sac" ,help="Path of the config file")
    parser.add_argument("--trainid", type=int, default=0 ,help="Train ID")
    parser.add_argument("--restart", type=bool, default=False ,help="Set true if you want to restart a training")
    parser.add_argument("--restart_epoch", type=int, default=5 ,help="The epoch number from where you want to restart the training.")
    # Example: python3 main.py --configfile /cfg/alma.yaml 0
    args = parser.parse_args()

    # Init logger ###############################################x
    current_dir = dirname(abspath(__file__))

    logger = Logger(current_dir = current_dir, main_args = args, light_mode = False)
    config = logger.get_config()
    
    # Init CUDA and torch and np ##################################
    init_cuda(config['hardware']['gpu'][args.trainid],config['hardware']['cpu_min'][args.trainid],config['hardware']['cpu_max'][args.trainid])

    print_torch_info(logger)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.print_logfile(device)

    torch.set_num_threads(torch.get_num_threads())

    torch.manual_seed(config['general']['seed'])
    np.random.seed(config['general']['seed'])

    
    replay_buffer = ReplayBuffer(
            obs_dim=int(config['environment']['obs_dim']), 
            act_dim=int(config['environment']['act_dim']), 
            size=int(config['buffer']['replay_buffer_size']))
    
    agent = Agent(0,device,config)

    samplerTrainerTester = SamplerTrainerTester(device,logger,config)

    samplerTrainerTester.start(agent,replay_buffer)



if __name__ == '__main__':
    main()

    
