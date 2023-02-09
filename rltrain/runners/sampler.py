import os
import math
import numpy as np
import torch
import time

from tqdm import tqdm

from rltrain.buffers.replay import ReplayBuffer
from rltrain.agents.agent import Agent

from rltrain.envs.builder import make_env

class Sampler:

    def __init__(self,agent,demo_buffer,config):
        #self.env = env
        self.agent = agent
        self.demo_buffer = demo_buffer
        self.config = config

        self.seed = config['general']['seed']        
        self.start_steps = config['sampler']['start_steps'] 
        self.max_ep_len = config['sampler']['max_ep_len'] 
        """
        Trainer

        Args:
            env : RLBench task-environment

            seed (int): Seed for random number generators.

            start_steps (int): Number of steps for uniform-random action selection,
                before running real policy. Helps exploration.

            max_ep_len (int): Maximum length of trajectory / episode / rollout.

        """
    
    def reset_env(self,sample2train):
        while True:
            try:
                o = self.env.reset_once()
                if self.env.init_state_valid():
                    break
                else:
                    data = {'code': -21, 'description':'Init state is not valid. Repeat env reset.'}
                    sample2train.put(data)
                    time.sleep(0.1)
            except:
                data = {'code': -1, 'description':'Could not reset the environment. Repeat env reset.'}
                sample2train.put(data)
                time.sleep(1)
        
        return o


    def start(self,id,replay_buffer,end_flag,pause_flag,sample2train):

        torch.manual_seed(self.seed*id)
        np.random.seed(self.seed*id)

        self.env = make_env(self.config)
        
        # Prepare for interaction with environment
        total_steps = 4

        o, ep_ret, ep_len = self.reset_env(sample2train), 0, 0

        # Main loop: collect experience in env and update/log each epoch
        #env_error_num = 0

        #pbar = tqdm(total = total_steps, desc =str(id) + ". sampler: ", colour="green")
        t = 0
        while end_flag.value:

            while pause_flag.value:
                time.sleep(0.1)

            # Until start_steps have elapsed, randomly sample actions
            # from a uniform distribution for better exploration. Afterwards, 
            # use the learned policy. 
            if t > self.start_steps:
                a = self.agent.get_action(o)
            else:
                a = self.agent.get_random_action()

            # Step the env
            try:
                o2, r, d, info = self.env.step(a)
            except:
                data = {'code': -2, 'description': 'Error in environment in step function, thus reseting the environment' + str(a)}
                sample2train.put(data)
                o, ep_ret, ep_len = self.reset_env(sample2train), 0, 0   
                continue     
            
            if bool(info): 
                if 'code' in info: 
                    sample2train.put(info)             
                    if info['code'] < 0:
                        o, ep_ret, ep_len = self.reset_env(sample2train), 0, 0
                        continue

            ep_ret += r
            ep_len += 1

            # Ignore the "done" signal if it comes from hitting the time
            # horizon (that is, when it's an artificial terminal signal
            # that isn't based on the agent's state)
            d = False if ep_len==self.max_ep_len else d

            # Store experience to replay buffer
            replay_buffer.store(o, a, r, o2, d)

            # Super critical, easy to overlook step: make sure to update 
            # most recent observation!
            o = o2

            # End of trajectory handling
            if d or (ep_len == self.max_ep_len):             
                o, ep_ret, ep_len = self.reset_env(sample2train), 0, 0               
          
            t += 1           
            #pbar.update(1)   
            #time.sleep(0.1)     

        self.env.shuttdown() 
     