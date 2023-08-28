from os.path import dirname, abspath

import numpy as np
import torch
import argparse

from rltrain.buffers.replay import ReplayBuffer

from rltrain.utils.utils import init_cuda, print_torch_info
from rltrain.logger.logger import Logger

from rltrain.agents.builder import make_agent

import datetime
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

import torch
import pandas as pd
    

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="cfg_exp/manual/config.yaml" ,help="Path of the config file")
    parser.add_argument("--trainid", type=int, default=0 ,help="Train ID")

    # Example: python3 main.py --configfile /cfg/alma.yaml 0
    args = parser.parse_args()

    # Init logger ###############################################x
    current_dir = dirname(abspath(__file__))

    logger = Logger(current_dir = current_dir, main_args = args, display_mode = False, tb_layout = False)
    config = logger.get_config()

    config_framework = logger.get_config_framework()
    
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
            size=int(float(config['buffer']['replay_buffer_size'])))
    
    agent = make_agent(device,config,config_framework)

    if config['trainer']['mode'] == "normal":
        from rltrain.runners.sampler_trainer_tester import SamplerTrainerTester
    elif config['trainer']['mode'] == "rs":
        from rltrain.runners.sampler_trainer_tester_rs import SamplerTrainerTester
    elif config['trainer']['mode'] == "cl":
        from rltrain.runners.sampler_trainer_tester_cl import SamplerTrainerTester
    
    samplerTrainerTester = SamplerTrainerTester(device,logger,config,args,config_framework)

    samplerTrainerTester.start(agent,replay_buffer)


if __name__ == '__main__':
    main()

    
