import os
from os.path import dirname, abspath

import numpy as np
import torch
import argparse
import rltrain.agents.sac.core as core

from rltrain.utils.utils import init_cuda, print_torch_info
from rltrain.logger.logger import Logger
from rltrain.runners.tester import Tester

from rltrain.envs.builder import make_env

import datetime
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

import torch
from tqdm import tqdm

def create_random_actions(config,config_framework,demo_buffer_size):

    env = make_env(config,config_framework)

    max_ep_len = config['sampler']['max_ep_len']
    # Boundary
    act_dim = config['environment']['act_dim']
    boundary_min = np.array(config['agent']['boundary_min'])[:act_dim]
    boundary_max = np.array(config['agent']['boundary_max'])[:act_dim]   

    pbar = tqdm(total=int(demo_buffer_size),colour="green")
    t = 0
    unsuccessful_num = 0
    while t<int(demo_buffer_size):

        ep_transitions = []
        ret = 0
        
        o, _ = env.reset_with_init_check()
    
        d = 0
        for _ in range(max_ep_len):

            a = np.random.uniform(low=boundary_min, high=boundary_max, size=act_dim)

            try:
                #time.sleep(0.1)
                o2, r, terminated, truncated, info = env.step(a)
                d = terminated or truncated
                #print(info)
                ep_transitions.append((o, a, r, o2, d))
                o = o2
                if d == 1:
                    break
            except:
                tqdm.write("Error in simulation, this demonstration is not added")
                d = 0
                break

            t+=1
            pbar.update(1)

            
    pbar.close()
    
    env.shuttdown()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="cfg_exp/manual/config.yaml" ,help="Path of the config file")
    parser.add_argument("--trainid", type=int, default=0 ,help="Train ID")
    args = parser.parse_args()

    # Init logger 
    current_dir = dirname(abspath(__file__))
    config_path = os.path.join(current_dir,args.config)

    logger = Logger(current_dir = current_dir, main_args = args, display_mode = False, tb_layout = False)
    config = logger.get_config()
    config_framework = logger.get_config_framework()

    # Init CUDA
    hw_train_id = 0
    init_cuda(config['hardware']['gpu'][hw_train_id],config['hardware']['cpu_min'][hw_train_id],config['hardware']['cpu_max'][hw_train_id])

    print_torch_info(logger)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    torch.set_num_threads(torch.get_num_threads())

    create_random_actions(config,config_framework,500)

    
