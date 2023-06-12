import os
from os.path import dirname, abspath

import numpy as np
import torch
import argparse
import rltrain.agents.core as core

from rltrain.utils.utils import init_cuda, print_torch_info
from rltrain.logger.logger import Logger
from rltrain.envs.RLBenchEnv import RLBenchEnv
from rltrain.runners.tester import Tester
from rltrain.agents.agent import Agent

import pandas as pd

import datetime
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

import torch

ENVS = ['rlbench', 'simsim','simsimv2','gym']

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--configfile", default="logs/0420_A_stack_blocks_sac/" ,help="Path of the config file")
    #parser.add_argument("--configfile", default="logs/0308_C_MountainCarContinuous-v0_sac/" ,help="Path of the config file")
    parser.add_argument("--trainid", type=int, default=0 ,help="Train ID")
    parser.add_argument("--restart", type=bool, default=False ,help="Set true if you want to restart a training")
    parser.add_argument("--test2env", default="simsimv2" ,help="Name of test2 env")
    # Example: python3 main.py --configfile /cfg/alma.yaml 0
    args = parser.parse_args()

    # Init logger 
    current_dir = dirname(abspath(__file__))

    args.restart = True
    logger = Logger(current_dir = current_dir, main_args = args)
    config = logger.get_config()

    # Init CUDA
    hw_train_id = 0
    init_cuda(config['hardware']['gpu'][hw_train_id],config['hardware']['cpu_min'][hw_train_id],config['hardware']['cpu_max'][hw_train_id])

    print_torch_info(logger)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Init RLBenchEnv

    print(device)

    torch.set_num_threads(torch.get_num_threads())

    if args.test2env not in ENVS:
        config['environment']['name'] = config['tester2']['env_name']
    else:
        config['environment']['name'] = args.test2env
    config['environment']['headless'] = True

    # Init Agent
    agent = Agent(0,device,config)

    # Init Trainer
    tester = Tester(agent,logger,config)

    # Test Agent
    avg_return, succes_rate, avg_episode_len, error_in_env, out_of_bounds = tester.test_agent_all(config['tester']['num_test_episodes'])

    d = {'avg_return': avg_return, 'succes_rate': succes_rate, 'avg_episode_len': avg_episode_len, 'error_in_env': error_in_env, 'out_of_bounds': out_of_bounds}
    df = pd.DataFrame(data=d)

    logger.save_test2(df,config['environment']['name'])

    print(df)