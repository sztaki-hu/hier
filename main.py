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
    
    
    env_num = int(config['sampler']['env_num'])
    if  env_num == 1: # 1 process #################

        logger = Logger(current_dir = current_dir, main_args = args, light_mode = False)
       
        env = make_env(config)

        sampler_trainer_tester = SamplerTrainerTester(device,env,demo_buffer,logger,config)

        try:
            # Start Training
            sampler_trainer_tester.start()
        finally:
            env.shuttdown()

    else:  # Multi-process ####################################################

        BaseManager.register('ReplayBuffer', ReplayBuffer)
        BaseManager.register('Agent', Agent)
        BaseManager.register('Logger', Logger)
        manager = BaseManager()
        manager.start()
        replay_buffer = manager.ReplayBuffer(
            obs_dim=int(config['environment']['obs_dim']), 
            act_dim=int(config['environment']['act_dim']), 
            size=int(config['buffer']['replay_buffer_size']))
        agent = manager.Agent(device,config)
        logger = manager.Logger(current_dir = current_dir, main_args = args, light_mode = False)

        # Training ##############################################################

        if args.restart == True: 
            path = logger.get_model_path_for_restart(dir = "model_backup_restart", epoch = int(args.restart_epoch))
            agent.load_weights(path)

            ## To be implemented #####################
            #replay_buffer = logger.load_replay_buffer(int(args.restart_epoch))
            ####################################

        sampler = Sampler(agent,demo_buffer,config)
        trainer = Trainer(device,demo_buffer,logger,config)
        
        end_flag = Value(c_bool, False)  
        pause_flag = Value(c_bool, True)
        t_limit = Value('i', 0)
        t_glob = Value('i', 0)
        test2train = mp.Queue()
        sample2train = mp.Queue(maxsize=100000)
        
        processes = []

        for i in range(env_num):
            p = Process(target=sampler.start, args=[i+1,replay_buffer,end_flag,pause_flag,sample2train,t_glob,t_limit])
            p.start()
            processes.append(p)
        
        agent_tester = Agent(device,config)
        tester = Tester(agent_tester,logger,config)

        tester_p = Process(target=tester.start, args=[end_flag,test2train])
        tester_p.start()
     

        trainer.start(agent,replay_buffer,pause_flag,test2train,sample2train,t_glob,t_limit)

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


if __name__ == '__main__':
    main()

    
