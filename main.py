from os.path import dirname, abspath

import numpy as np
import torch
import time
import argparse

import multiprocessing as mp
from multiprocessing import Process, Value
from multiprocessing.managers import BaseManager
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
from rltrain.runners.tester import Tester

import datetime
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

import torch
import pandas as pd
    

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--configfile", default="cfg/config.yaml" ,help="Path of the config file")
    #parser.add_argument("--configfile", default="logs/0216_A_test_stack_blocks_sac" ,help="Path of the config file")
    parser.add_argument("--trainid", type=int, default=0 ,help="Train ID")
    parser.add_argument("--restart", type=bool, default=False ,help="Set true if you want to restart a training")
    parser.add_argument("--restart_epoch", type=int, default=5 ,help="The epoch number from where you want to restart the training.")
    # Example: python3 main.py --configfile /cfg/alma.yaml 0
    args = parser.parse_args()

    # Init logger ###############################################x
    current_dir = dirname(abspath(__file__))

    logger = Logger(current_dir = current_dir, main_args = args, light_mode = True)
    config = logger.get_config()
    
    # Init CUDA and torch and np ##################################
    init_cuda(config['hardware']['gpu'][args.trainid],config['hardware']['cpu_min'][args.trainid],config['hardware']['cpu_max'][args.trainid])

    print_torch_info(logger)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.print_logfile(device)

    torch.set_num_threads(torch.get_num_threads())

    torch.manual_seed(config['general']['seed'])
    np.random.seed(config['general']['seed'])

    mp.set_start_method('spawn')
    #torch.multiprocessing.set_start_method('spawn')

    # Demo Buffer ###########################################################
    if config['demo']['demo_use']:
        demo = Demo(logger,config)
        logger.print_logfile("Waiting for demos...")  
        while demo.demo_exists() == False:
            time.sleep(1.0)     
        demo_buffer = demo.load_demos()
        logger.print_logfile("Demos are loaded")  
    else:
        demo_buffer = None
    
    
    sampler_num = int(config['sampler']['sampler_num'])
    agent_num = int(config['agent']['agent_num'])

    BaseManager.register('ReplayBuffer', ReplayBuffer)
    BaseManager.register('Logger', Logger)
    BaseManager.register('Agent', Agent)
      
    manager = BaseManager()
    manager.start()

    buffer_num = int(config['buffer']['buffer_num'])
    replay_buffers = []

    for _ in range(buffer_num):
        replay_buffers.append(manager.ReplayBuffer(
            obs_dim=int(config['environment']['obs_dim']), 
            act_dim=int(config['environment']['act_dim']), 
            size=int(config['buffer']['replay_buffer_size'])))

    logger = manager.Logger(current_dir = current_dir, main_args = args, light_mode = False)

    agents = []
    for agent_id in range(agent_num):
        agents.append(manager.Agent(agent_id,device,config))

    sampler = Sampler(demo_buffer,config)
    trainer = Trainer(device,demo_buffer,logger,config)
        
    end_flag = Value(c_bool, False)  
    pause_flag = Value(c_bool, True)
    t_limit = Value('i', -int(config['trainer']['update_after']))
    t_glob = Value('i', 0)
    test2train = mp.Queue()
    sample2train = mp.Queue(maxsize=100000)

    processes = []

    for agent_id in range(agent_num):
        for sampler_id in range(sampler_num):
            p = Process(target=sampler.start, args=[agent_id,agents[agent_id],sampler_id,replay_buffers[agent_id%buffer_num],end_flag,pause_flag,sample2train,t_glob,t_limit])
            p.start()
            processes.append(p)
    
    agent_tester = Agent(0,device,config)
    tester = Tester(agent_tester,logger,config)

    tester_p = Process(target=tester.start, args=[end_flag,test2train])
    tester_p.start()

    trainer.start(agents,replay_buffers,pause_flag,test2train,sample2train,t_glob,t_limit)

    # Stop Training #############################################################

    pause_flag.value = False
    end_flag.value = True

    logger.print_logfile("Waiting for samplers to terminate")
    for p in processes:
        p.join()
    logger.print_logfile("Samplers terminated")

    logger.print_logfile("Waiting for the tester to terminate")
    tester_p.join()
    logger.print_logfile("Tester terminated")

    while sample2train.empty() == False:
        logger.print_logfile(sample2train.get())
    
    while test2train.empty() == False:
        logger.print_logfile(test2train.get())
    

    # Test2 #######################################################################

    if config['tester2']['bool'] == True:
        # Init RLBenchEnv
        config['environment']['name'] = config['tester2']['env_name']
        config['environment']['headless'] = True

        # Init Agent
        agent = Agent(0,device,config)

        # Init Trainer
        tester = Tester(agent,logger,config)

        # Test Agent
        avg_return, succes_rate, avg_episode_len, error_in_env, out_of_bounds = tester.test_agent_all(config['tester2']['num_test2_episodes'])

        d = {'avg_return': avg_return, 'succes_rate': succes_rate, 'avg_episode_len': avg_episode_len, 'error_in_env': error_in_env, 'out_of_bounds': out_of_bounds}
        df = pd.DataFrame(data=d)

        logger.save_test2(df)

        print(df)


if __name__ == '__main__':
    main()

    
