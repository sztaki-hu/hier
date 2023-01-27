from os.path import dirname, abspath

import numpy as np
import torch
from tqdm import tqdm
import time

import multiprocessing as mp
from multiprocessing import Process, Value, Manager
from multiprocessing.managers import BaseManager
from multiprocessing.managers import SyncManager, MakeProxyType, public_methods
from ctypes import c_bool
import torch.nn as nn

from rltrain.agents.agent import Agent
from rltrain.buffers.demo import Demo
from rltrain.buffers.replay import ReplayBuffer

from rltrain.utils.utils import init_cuda, print_torch_info
from rltrain.logger.logger import Logger
from rltrain.envs.builder import make_env
from rltrain.runners.sampler_trainer_tester import SamplerTrainerTester
from rltrain.runners.sampler import Sampler
from rltrain.runners.trainer import Trainer
from rltrain.agents.core import SquashedGaussianMLPActor

import datetime
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

import torch
    

def main():

    # Init logger ###############################################x
    current_dir = dirname(abspath(__file__))
    config_path = current_dir + "/cfg/config.yaml"
    #config_path = current_dir + "/cfg/config_test.yaml"
    logger = Logger(current_dir = current_dir, config_path = config_path)
    config = logger.get_config()
    
    # Init CUDA and torch and np ##################################
    init_cuda(config['hardware']['gpu'],config['hardware']['cpu_min'],config['hardware']['cpu_max'])

    print_torch_info()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    torch.set_num_threads(torch.get_num_threads())

    torch.manual_seed(config['general']['seed'])
    np.random.seed(config['general']['seed'])

    # Init Demo Buffer ################################################
    if config['demo']['demo_use']:
        demo = Demo(logger,config)  
        if demo.demo_create:
            demo_p = Process(target=demo.create_demos, args=[])
            demo_p.start()
            demo_p.join()
        demo_buffer = demo.load_demos()
    else:
        demo_buffer = None

    env_num = int(config['sampler']['env_num'])
    if  env_num == 1: # 1 process #################
       
        env = make_env(config)

        sampler_trainer_tester = SamplerTrainerTester(device,env,demo_buffer,logger,config)

        try:
            # Start Training
            sampler_trainer_tester.start()
        finally:
            env.shuttdown()

    else:  # Multi-process ##############################################

        mp.set_start_method('spawn')
        #torch.multiprocessing.set_start_method('spawn')

        BaseManager.register('ReplayBuffer', ReplayBuffer)
        BaseManager.register('Agent', Agent)
        manager = BaseManager()
        manager.start()
        replay_buffer = manager.ReplayBuffer(
            obs_dim=int(config['environment']['obs_dim']), 
            act_dim=int(config['environment']['act_dim']), 
            size=int(config['buffer']['replay_buffer_size']))
        agent = manager.Agent(device,config)

        sampler = Sampler(agent,demo_buffer,config)
        trainer = Trainer(device,demo_buffer,logger,config)
        
        end_flag = Value(c_bool, True)  
        pause_flag = Value(c_bool, True)
        env_error_num = Value('i',0)
        #print(end_flag.value)
        
        processes = []

        for i in range(env_num):
            p = Process(target=sampler.start, args=[i+1,replay_buffer,end_flag,pause_flag,env_error_num])
            p.start()
            processes.append(p)
        
        trainer.start(agent,replay_buffer,pause_flag,env_error_num)

        pause_flag.value = False
        end_flag.value = False

        print("Wait for processes to terminate")
        for p in processes:
            p.join()
        print("Processes terminated")


if __name__ == '__main__':
    main()

    
