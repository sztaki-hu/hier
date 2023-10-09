import os
from os.path import dirname, abspath

import numpy as np
import torch
import argparse

from rltrain.buffers.replay import ReplayBuffer

from rltrain.utils.utils import init_cuda, print_torch_info
from rltrain.logger.logger import Logger

from rltrain.agents.builder import make_agent
from rltrain.runners.sampler_trainer_tester import SamplerTrainerTester

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
    parser.add_argument("--config", default="cfg_exp/auto/config.yaml", help="Path of the config file")
    parser.add_argument("--explist", default="cfg_exp/auto/exp_list.yaml", help="Path of the config file")
    parser.add_argument("--hwid", type=int, default=0, help="Hardware id")
    parser.add_argument("--seednum", type=int, default=3, help="seednum")
    parser.add_argument("--processid", type=int, default=0, help="processid")
    parser.add_argument("--testconfig", type=bool, default=True, help="Test config file")
    args = parser.parse_args()

    # Get experiments
    current_dir = dirname(abspath(__file__))
    exp_lists = load_yaml(os.path.join(current_dir,args.explist))
    exp_list = exp_lists['process_'+str(args.processid)]

    # Agents
    agents = exp_list['agent']['type']
    # Envs
    reward_bonuses = exp_list['environment']['reward']['reward_bonus']
    # Buffers
    her_strategies = exp_list['buffer']['her']['goal_selection_strategy']
    replay_buffer_sizes = exp_list['buffer']['replay_buffer_size']
    highlights_batch_ratios = exp_list['buffer']['highlights']['batch_ratio']
    # Trainers
    trainer_set_of_total_timesteps = exp_list['trainer']['total_timesteps']
    # Eval
    eval_freqs = exp_list['eval']['freq']
    # Tasks
    envs = list(exp_list['task'].keys())
    # CLs
    cl_types = exp_list['cl']['type']
    cl_range_growth_modes = exp_list['cl']['range_growth_mode']

    config_file_is_valid = True
    error_exps = []

    test_list = [True, False] if args.testconfig else [False] 

    for is_test_config in test_list:
        
        seednum = 1 if is_test_config else args.seednum

        if is_test_config == False:
            if config_file_is_valid == False:
                print("##################################################################")
                print("                  Config file is not valid!")
                for error_exp in error_exps:
                    print("---------------------------------------------------------------")
                    print(error_exp)
                print("##################################################################")
                return -1
            else:
                print("##################################################################")
                print("                    Config file is valid!")
                print("##################################################################")        
        
        for _ in range(seednum):
            for env_name in envs:
                for task_name in exp_list['task'][env_name]:
                    for agent_type in agents:
                        for her_strategy in her_strategies:
                            for cl_type in cl_types:
                                for cl_range_growth_mode in cl_range_growth_modes:
                                    for replay_buffer_size in replay_buffer_sizes:
                                        for reward_bonus in reward_bonuses:
                                            for highlights_batch_ratio in highlights_batch_ratios:
                                                for trainer_total_timesteps in trainer_set_of_total_timesteps:   
                                                    for eval_freq in eval_freqs:   
                                                        try:                                  
                                                            exp = {}
                                                            exp['main'] = {} 
                                                            exp['exp_in_name'] = {}
                                                            exp['exp_abb'] = {}
                                                            exp['main']['env'] = env_name
                                                            exp['exp_in_name']['env'] = False
                                                            exp['main']['task'] = task_name
                                                            exp['exp_in_name']['task'] = True
                                                            exp['main']['agent'] = agent_type
                                                            exp['exp_in_name']['agent'] = True
                                                            exp['main']['her_strategy'] = her_strategy
                                                            exp['exp_in_name']['her_strategy'] = False
                                                            exp['main']['cl'] = cl_type
                                                            exp['exp_in_name']['cl'] = True
                                                            exp['main']['cl_range_growth_mode'] = cl_range_growth_mode
                                                            exp['exp_in_name']['cl_range_growth_mode'] = False
                                                            exp['main']['replay_buffer_size'] = replay_buffer_size
                                                            exp['exp_in_name']['replay_buffer_size'] = False
                                                            exp['main']['reward_bonus'] = reward_bonus
                                                            exp['exp_in_name']['reward_bonus'] = False
                                                            exp['exp_abb']['reward_bonus'] = 'rb'
                                                            exp['main']['highlights_batch_ratio'] = highlights_batch_ratio
                                                            exp['exp_in_name']['highlights_batch_ratio'] = False
                                                            exp['exp_abb']['highlights_batch_ratio'] = 'hbr'
                                                            exp['main']['trainer_total_timesteps'] = trainer_total_timesteps
                                                            exp['exp_in_name']['trainer_total_timesteps'] = False
                                                            exp['main']['eval_freq'] = eval_freq
                                                            exp['exp_in_name']['eval_freq'] = False

                                                            # Init logger ###############################################x
                                                            logger = Logger(current_dir = current_dir, main_args = args, display_mode = False, exp = exp, is_test_config = is_test_config)

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

                                                            samplerTrainerTester = SamplerTrainerTester(device,logger,config,args,config_framework,replay_buffer,agent)

                                                            if is_test_config == False: 
                                                                samplerTrainerTester.start()

                                                        except:
                                                            config_file_is_valid = False
                                                            error_exps.append(exp)
                                                            print("Error")

    if config_file_is_valid == False:
            print("##################################################################")
            print("                  Errors were raised:")
            for error_exp in error_exps:
                print("---------------------------------------------------------------")
                print(error_exp)
            print("##################################################################")
            return -1
    else:
        print("##################################################################")
        print("                    No errors!")
        print("##################################################################")     


if __name__ == '__main__':
    main()