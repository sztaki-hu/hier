import os
from os.path import dirname, abspath

import numpy as np
import torch
import argparse

from rltrain.buffers.replay_v0 import ReplayBuffer

from rltrain.utils.utils import init_cuda, print_torch_info
from rltrain.logger.logger import Logger

from rltrain.agents.builder import make_agent
from rltrain.runners.uniprocess.sampler_trainer_tester_v2 import SamplerTrainerTester


import datetime
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

import torch
import pandas as pd
import yaml


def load_yaml(file):
    if file is not None:
        with open(file) as f:
            return yaml.load(f, Loader=yaml.UnsafeLoader)
    return None
    

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="cfg_exp/auto/config.yaml" ,help="Path of the config file")
    parser.add_argument("--explist", default="cfg_exp/auto/exp_list.yaml" ,help="Path of the config file")
    parser.add_argument("--trainid", type=int, default=0 ,help="Train ID")
    args = parser.parse_args()

    # Get experiments
    current_dir = dirname(abspath(__file__))
    exp_list = load_yaml(os.path.join(current_dir,args.explist))

    agents = exp_list['agent_list']
    envs = list(exp_list['task_list'].keys())

    for env in envs:
        for task in exp_list['task_list'][env]:
            for agent in agents:
                exp = {}
                exp['env'] = env
                exp['task'] = task
                exp['agent'] = agent
                

                # Init logger ###############################################x
                logger = Logger(current_dir = current_dir, main_args = args, display_mode = False, tb_layout = False, exp = exp)

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
                        size=int(config['buffer']['replay_buffer_size']))
                
                agent = make_agent(0,device,config,config_framework)

                samplerTrainerTester = SamplerTrainerTester(device,logger,config,args,config_framework)

                samplerTrainerTester.start(agent,replay_buffer)

if __name__ == '__main__':
    main()