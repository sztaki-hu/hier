import os
from os.path import dirname, abspath

import numpy as np
import torch
import argparse
import rltrain.agents.sac.core as core

from rltrain.utils.utils import init_cuda, print_torch_info
from rltrain.logger.logger import Logger
from rltrain.envs.RLBenchEnv import RLBenchEnv
from rltrain.utils.eval import Eval

from rltrain.agents.builder import make_agent

import datetime
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

import torch

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="logs/0928_A_PandaSlide-v3_sac_controldiscrete_const" ,help="Path of the config file")
    parser.add_argument("--hwid", type=int, default=0 ,help="Hardware id")
    parser.add_argument("--seedid", type=int, default=5 ,help="seedid") #1
    # Example: python3 main.py --configfile /cfg/alma.yaml 0
    args = parser.parse_args()

    # Init logger 
    current_dir = dirname(abspath(__file__))

    logger = Logger(current_dir = current_dir, main_args = args, display_mode = True)
    config = logger.get_config()
    config_framework = logger.get_config_framework()

    # Init CUDA
    init_cuda(config['hardware']['gpu'][args.hwid],config['hardware']['cpu_min'][args.hwid],config['hardware']['cpu_max'][args.hwid])

    print_torch_info(logger)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    torch.set_num_threads(torch.get_num_threads())

    # Init Agent
    agent = make_agent(device,config,config_framework)

    # Init Trainer
    tester = Eval(agent,logger,config,config_framework)

    # Test Agent
    tester.eval_agent(model_name="best_model",num_display_episode=20, headless=False, time_delay = 0.05, current_dir = current_dir)


    
