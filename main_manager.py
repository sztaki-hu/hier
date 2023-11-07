import os
from os.path import dirname, abspath
import argparse
import itertools
from typing import Dict
import datetime
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
import yaml


def load_yaml(file: str) -> Dict:
    if file is not None:
        with open(file) as f:
            return yaml.load(f, Loader=yaml.UnsafeLoader)
    return {}

def save_yaml(path: str, data: Dict) -> None:
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)

def create_folder(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)
        print(path + ' folder is created!')
    else:
        print(path + ' folder already exists!')

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="cfg_exp/auto/config.yaml", help="Path of the config file")
    parser.add_argument("--explist", default="cfg_exp/auto/exp_list.yaml", help="Path of the config file")
    parser.add_argument("--processid", type=int, default=0, help="processid")
    parser.add_argument("--testconfig", type=bool, default=True, help="Test config file")
    parser.add_argument("--tempconfig", default="cfg_exp/auto/temp", help="Path of the dir of temp config")
    args = parser.parse_args()

    hwid_list = [0,1,2,3]
    hw_i = 0

    

    # Get experiments
    current_dir = dirname(abspath(__file__))
    exp_lists = load_yaml(os.path.join(current_dir,args.explist))
    exp_list = exp_lists['process_'+str(args.processid)]

    create_folder(os.path.join(current_dir,args.tempconfig))

    # General
    seednum = exp_list['general']['seednum']
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
    # CLs
    cl_types = exp_list['cl']['type']
    cl_range_growth_modes = exp_list['cl']['range_growth_mode']

    iter = 0

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
                    # CL
                    cl_types,
                    cl_range_growth_modes,
                    ):

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
                # CL
                cl_type                                  = r[19]
                cl_range_growth_mode                     = r[20]                  
                                        
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
                # CL
                exp['main']['cl'] = cl_type
                exp['exp_in_name']['cl'] = True
                exp['main']['cl_range_growth_mode'] = cl_range_growth_mode
                exp['exp_in_name']['cl_range_growth_mode'] = False


                timetag = "_".join([timestamp, str(iter)])  
                exppath = os.path.join(current_dir,args.tempconfig,timetag + "_temp_exp_config.yaml")        
                save_yaml(exppath,exp)
                iter += 1    

                command = "python3 main.py --config " + args.config + " --hwid " + str(hwid_list[hw_i]) + " --seednum " + str(seednum) + " --exppath " + str(exppath) + "&"
                #print(command)           
                os.system(command)   

                hw_i += 1 if hw_i < len(hwid_list)-1 else 0 

                        

if __name__ == '__main__':
    main()