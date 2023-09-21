import os
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
    parser.add_argument("--hwid", type=int, default=0 ,help="Hardware id")
    parser.add_argument("--seednum", type=int, default=3 ,help="seednum")
    parser.add_argument("--processid", type=int, default=0 ,help="processid")
    args = parser.parse_args()

    # Get experiments
    current_dir = dirname(abspath(__file__))
    exp_lists = load_yaml(os.path.join(current_dir,args.explist))
    exp_list = exp_lists['process_'+str(args.processid)]

    agents = exp_list['agents']
    envs = list(exp_list['tasks'].keys())
    her_strategies = exp_list['her_strategies']
    cl_types = exp_list['cl']

    for env_name in envs:
        for task_name in exp_list['tasks'][env_name]:
            for agent_type in agents:
                for her_strategy in her_strategies:
                    for cl_type in cl_types:
                        for _ in range(args.seednum):
                            exp = {}
                            exp['env'] = env_name
                            exp['task'] = task_name
                            exp['agent'] = agent_type
                            exp['her_strategy'] = her_strategy
                            exp['cl'] = cl_type

                            # Init logger ###############################################x
                            logger = Logger(current_dir = current_dir, main_args = args, display_mode = False, exp = exp)

                            config = logger.get_config()

                            config_framework = logger.get_config_framework()
                            
                            # Init CUDA and torch and np ##################################
                            init_cuda(config['hardware']['gpu'][args.hwid],config['hardware']['cpu_min'][args.hwid],config['hardware']['cpu_max'][args.hwid])

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