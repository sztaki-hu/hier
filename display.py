import os
from os.path import dirname, abspath

import numpy as np
import torch
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
    # Init logger
    trainid = 0
    current_dir = dirname(abspath(__file__))
    config_path = current_dir + "/logs_compute/0206_C_stack_blocks_sac/"+ str(trainid) +"/config.yaml"
    logger = Logger(current_dir = current_dir, config_path = config_path, trainid = trainid)
    config = logger.get_config()

    # Init CUDA
    hw_train_id = 0
    init_cuda(config['hardware']['gpu'][hw_train_id],config['hardware']['cpu_min'][hw_train_id],config['hardware']['cpu_max'][hw_train_id])

    print_torch_info()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    torch.set_num_threads(torch.get_num_threads())

    # Init RLBenchEnv
    config['environment']['name'] = 'rlbench'
    config['environment']['headless'] = False

    # Init Agent
    agent = Agent(device,config)

    # Init Trainer
    tester = Tester(agent,logger,config)

    # Test Agent
    tester.display_agent("model_3",5)

    # Test Agent all models
    # models = logger.list_model_dir()
    # models = sorted(models)
    # for model in models:
    #     tester.display_agent(model,3)


    
