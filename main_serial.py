import os
from os.path import dirname, abspath

import numpy as np
import torch
import argparse
import itertools
from typing import Dict, Union, Optional

from rltrain.utils.utils import init_cuda, print_torch_info
from rltrain.logger.logger import Logger

from rltrain.runners.sampler_trainer_tester import SamplerTrainerTester

import datetime
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

import torch
import pandas as pd
import yaml


def load_yaml(file: str) -> Dict:
    if file is not None:
        with open(file) as f:
            return yaml.load(f, Loader=yaml.UnsafeLoader)
    return {}

def main() -> int:

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="cfg_exp/multi/config.yaml", help="Path of the config file")
    parser.add_argument("--explist", default="cfg_exp/multi/exp_list.yaml", help="Path of the config file")
    parser.add_argument("--hwid", type=int, default=0, help="Hardware id")
    parser.add_argument("--processid", type=int, default=0, help="processid")
    parser.add_argument("--testconfig", type=bool, default=True, help="Test config file")
    args = parser.parse_args()

    # Get experiments
    current_dir = dirname(abspath(__file__))
    exp_lists = load_yaml(os.path.join(current_dir,args.explist))
    exp_list = exp_lists['process_'+str(args.processid)]

    # General
    exp_seednum = exp_list['general']['seednum']
    # Agents
    agents = exp_list['agent']['type']
    agent_sac_alphas = exp_list['agent']['sac']['alpha']
    agent_gammas = exp_list['agent']['gamma']
    agent_learning_rates = exp_list['agent']['learning_rate']
    # Envs
    reward_shaping_types = exp_list['environment']['reward']['reward_shaping_type']
    reward_bonuses = exp_list['environment']['reward']['reward_bonus']
    # Buffers
    replay_buffer_sizes = exp_list['buffer']['replay_buffer_size']
    her_strategies = exp_list['buffer']['her']['goal_selection_strategy']
    hier_buffer_sizes = exp_list['buffer']['hier']['buffer_size']
    hier_lambda_modes = exp_list['buffer']['hier']['lambda']['mode']
    hier_lambda_fix_lambdas = exp_list['buffer']['hier']['lambda']['fix']['lambda']
    hier_lambda_predefined_lambda_starts = exp_list['buffer']['hier']['lambda']['predefined']['lambda_start']
    hier_lambda_predefined_lambda_ends = exp_list['buffer']['hier']['lambda']['predefined']['lambda_end']
    hier_xi_modes = exp_list['buffer']['hier']['xi']['mode']
    hier_xi_xis = exp_list['buffer']['hier']['xi']['xi']
    per_modes = exp_list['buffer']['per']['mode']
    
    # Trainers
    trainer_set_of_total_timesteps = exp_list['trainer']['total_timesteps']
    # Eval
    eval_freqs = exp_list['eval']['freq']
    eval_num_episodes_list = exp_list['eval']['num_episodes']
    # Tasks
    envs = list(exp_list['task'].keys())
    # ISE
    ise_types = exp_list['ise']['type']
    ise_range_growth_modes = exp_list['ise']['range_growth_mode']

    config_file_is_valid = True
    error_exps = []

    test_list = [True, False] if args.testconfig else [False] 

    for is_test_config in test_list:
        
        seednum = 1 if is_test_config else exp_seednum

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
                #return 0       
        
        for _ in range(seednum):
             for env_name in envs:
                for task_name in exp_list['task'][env_name]:
                    for r in itertools.product(
                            # Agents
                            agents, 
                            agent_sac_alphas,
                            agent_gammas,
                            agent_learning_rates,
                            # Envs
                            reward_shaping_types,
                            reward_bonuses,
                            # Buffers
                            replay_buffer_sizes,
                            her_strategies,
                            hier_buffer_sizes,
                            hier_lambda_modes,
                            hier_lambda_fix_lambdas,
                            hier_lambda_predefined_lambda_starts,
                            hier_lambda_predefined_lambda_ends,
                            hier_xi_modes,
                            hier_xi_xis,
                            per_modes,
                            # Trainers
                            trainer_set_of_total_timesteps,
                            # Eval
                            eval_freqs,
                            eval_num_episodes_list,
                            # ISE
                            ise_types,
                            ise_range_growth_modes,
                            ):
                        
                        #try:   

                        print(r)

                        # Agent
                        agent_type                               = r[0]
                        agent_sac_alpha                          = r[1]
                        agent_gamma                              = r[2]
                        agent_learning_rate                      = r[3]
                        # Env
                        reward_shaping_type                      = r[4]
                        reward_bonus                             = r[5]
                        # Buffer
                        replay_buffer_size                       = r[6]
                        her_strategy                             = r[7]
                        hier_buffer_size                         = r[8]
                        hier_lambda_mode                         = r[9]
                        hier_lambda_fix_lambda                   = r[10]
                        hier_lambda_predefined_lambda_start      = r[11]
                        hier_lambda_predefined_lambda_end        = r[12]
                        hier_xi_mode                             = r[13]
                        hier_xi_xi                               = r[14]
                        per_mode                                 = r[15]
                        # Trainer
                        trainer_total_timesteps                  = r[16]
                        # Eval
                        eval_freq                                = r[17]
                        eval_num_episodes                        = r[18]
                        # ISE
                        ise_type                                  = r[19]
                        ise_range_growth_mode                     = r[20]                  
                                                
                        exp = {}
                        exp['main'] = {} 
                        exp['exp_in_name'] = {}
                        exp['exp_abb'] = {}
                        # Task
                        exp['main']['env'] = env_name
                        exp['exp_in_name']['env'] = False
                        exp['main']['task'] = task_name
                        exp['exp_in_name']['task'] = True
                        # Agent
                        exp['main']['agent'] = agent_type
                        exp['exp_in_name']['agent'] = True
                        exp['main']['agent_sac_alpha'] = agent_sac_alpha
                        exp['exp_in_name']['agent_sac_alpha'] = False
                        exp['exp_abb']['agent_sac_alpha'] = 'alp'
                        exp['main']['agent_gamma'] = agent_gamma
                        exp['exp_in_name']['agent_gamma'] = False
                        exp['exp_abb']['agent_gamma'] = 'gam'
                        exp['main']['agent_learning_rate'] = agent_learning_rate
                        exp['exp_in_name']['agent_learning_rate'] = False
                        exp['exp_abb']['agent_learning_rate'] = 'lr'
                        # Env
                        exp['main']['reward_shaping_type'] = reward_shaping_type
                        exp['exp_in_name']['reward_shaping_type'] = True
                        exp['main']['reward_bonus'] = reward_bonus
                        exp['exp_in_name']['reward_bonus'] = False
                        exp['exp_abb']['reward_bonus'] = 'rb'
                        # Buffer
                        exp['main']['replay_buffer_size'] = replay_buffer_size
                        exp['exp_in_name']['replay_buffer_size'] = False                        
                        exp['main']['her_strategy'] = her_strategy
                        exp['exp_in_name']['her_strategy'] = True
                        exp['main']['hier_buffer_size'] = hier_buffer_size
                        exp['exp_in_name']['hier_buffer_size'] = False
                        exp['main']['hier_lambda_mode'] = hier_lambda_mode
                        exp['exp_in_name']['hier_lambda_mode'] = True
                        exp['main']['hier_lambda_fix_lambda'] = hier_lambda_fix_lambda
                        exp['exp_in_name']['hier_lambda_fix_lambda'] = False
                        exp['main']['hier_lambda_predefined_lambda_start'] = hier_lambda_predefined_lambda_start
                        exp['exp_in_name']['hier_lambda_predefined_lambda_start'] = False
                        exp['main']['hier_lambda_predefined_lambda_end'] = hier_lambda_predefined_lambda_end
                        exp['exp_in_name']['hier_lambda_predefined_lambda_end'] = False
                        exp['main']['hier_xi_mode'] = hier_xi_mode
                        exp['exp_in_name']['hier_xi_mode'] = True
                        exp['main']['hier_xi_xi'] = hier_xi_xi
                        exp['exp_in_name']['hier_xi_xi'] = False
                        exp['exp_abb']['hier_xi_xi'] = 'xi'
                        exp['main']['per_mode'] = per_mode
                        exp['exp_in_name']['per_mode'] = True
                        # Trainer
                        exp['main']['trainer_total_timesteps'] = trainer_total_timesteps
                        exp['exp_in_name']['trainer_total_timesteps'] = False
                        # Eval
                        exp['main']['eval_freq'] = eval_freq
                        exp['exp_in_name']['eval_freq'] = False
                        exp['main']['eval_num_episodes'] = eval_num_episodes
                        exp['exp_in_name']['eval_num_episodes'] = False
                        # ISE
                        exp['main']['ise'] = ise_type
                        exp['exp_in_name']['ise'] = True
                        exp['main']['ise_range_growth_mode'] = ise_range_growth_mode
                        exp['exp_in_name']['ise_range_growth_mode'] = False
                    

                        # Init logger ###############################################x
                        logger = Logger(current_dir = current_dir, configpath = args.config, display_mode = False, exp = exp, is_test_config = is_test_config)

                        config = logger.get_config()

                        config_framework = logger.get_config_framework()
                        
                        # Init CUDA and torch and np ##################################
                        init_cuda(config['hardware']['gpu'][args.hwid],config['hardware']['cpu_min'][args.hwid],config['hardware']['cpu_max'][args.hwid])

                        if is_test_config == False: print_torch_info(logger)

                        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                        if is_test_config == False: logger.print_logfile(str(device))

                        torch.set_num_threads(torch.get_num_threads())

                        torch.manual_seed(config['general']['seed'])
                        np.random.seed(config['general']['seed'])  

                        samplerTrainerTester = SamplerTrainerTester(device,logger,config,config_framework,)

                        if is_test_config == False: 
                            samplerTrainerTester.start()

                        # except:
                        #     config_file_is_valid = False
                        #     error_exps.append(exp)
                        #     print("Error")

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

    return 1  


if __name__ == '__main__':
    main()