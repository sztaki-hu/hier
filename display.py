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

import datetime
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

import torch

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--configfile", default="logs/0307_test_01_MountainCarContinuous-v0_sac/" ,help="Path of the config file")
    parser.add_argument("--trainid", type=int, default=0 ,help="Train ID")
    parser.add_argument("--restart", type=bool, default=False ,help="Set true if you want to restart a training")
    # Example: python3 main.py --configfile /cfg/alma.yaml 0
    args = parser.parse_args()

    # Init logger 
    current_dir = dirname(abspath(__file__))
    #config_path = current_dir + "/logs/0216_B_stack_blocks_sac/"+ str(trainid) +"/config.yaml"
    args.restart = True
    logger = Logger(current_dir = current_dir, main_args = args)
    config = logger.get_config()

    # Init CUDA
    hw_train_id = 0
    init_cuda(config['hardware']['gpu'][hw_train_id],config['hardware']['cpu_min'][hw_train_id],config['hardware']['cpu_max'][hw_train_id])

    print_torch_info(logger)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    torch.set_num_threads(torch.get_num_threads())

    # Init RLBenchEnv
    if config['environment']['name'] == 'simsim': config['environment']['name'] = 'rlbench' 
    config['environment']['headless'] = False

    # Init Agent
    agent = Agent(device,config)

    # Init Trainer
    tester = Tester(agent,logger,config)

    # Test Agent
    tester.display_agent("model_1",50)

    # Test Agent all models
    # models = logger.list_model_dir()
    # models = sorted(models)
    # for model in models:
    #     tester.display_agent(model,3)


    
