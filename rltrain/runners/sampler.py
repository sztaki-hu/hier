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

        self.mode_sync = config['general']['sync'] 

        self.seed = config['general']['seed']        
        self.start_steps = config['sampler']['start_steps'] 
        self.max_ep_len = config['sampler']['max_ep_len'] 
        self.gamma = config['agent']['gamma'] 
        self.n_step = config['agent']['n_step'] 

        self.heatmap_bool = config['logger']['heatmap']['bool']
        self.heatmap_res = config['logger']['heatmap']['resolution']

        self.act_dim = config['environment']['act_dim']
        self.boundary_min = np.array(config['agent']['boundary_min'])[:self.act_dim]
        self.boundary_max = np.array(config['agent']['boundary_max'])[:self.act_dim]

        if self.heatmap_bool:
            self.heatmap_pick = np.zeros((self.heatmap_res, self.heatmap_res))
            self.heatmap_place = np.zeros((self.heatmap_res, self.heatmap_res))
            self.bins = []
            for i in range(self.act_dim):
                self.bins.append(np.linspace(self.boundary_min[i], self.boundary_max[i], num=self.heatmap_res+1, retstep=False))

        """
        Sampler

        Args:
            env : RLBench task-environment

            seed (int): Seed for random number generators.

            start_steps (int): Number of steps for uniform-random action selection,
                before running real policy. Helps exploration.

            max_ep_len (int): Maximum length of trajectory / episode / rollout.

        """

    def update_heatmap(self,a):

        if self.act_dim == 6:
            pick_x = min(max(np.digitize(a[0], self.bins[0]) - 1,0),self.heatmap_res - 1)
            pick_y = min(max(np.digitize(a[1], self.bins[1]) - 1,0),self.heatmap_res - 1)
            place_x = min(max(np.digitize(a[3], self.bins[3]) - 1,0),self.heatmap_res - 1)
            place_y = min(max(np.digitize(a[4], self.bins[4]) - 1,0),self.heatmap_res - 1)
        elif self.act_dim == 4:
            pick_x = min(max(np.digitize(a[0], self.bins[0]) - 1,0),self.heatmap_res - 1)
            pick_y = min(max(np.digitize(a[1], self.bins[1]) - 1,0),self.heatmap_res - 1)
            place_x = min(max(np.digitize(a[2], self.bins[2]) - 1,0),self.heatmap_res - 1)
            place_y = min(max(np.digitize(a[3], self.bins[3]) - 1,0),self.heatmap_res - 1)

        self.heatmap_pick[pick_x][pick_y] += 1
        self.heatmap_place[place_x][place_y] += 1
    
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


    def start(self,id,replay_buffer,end_flag,pause_flag,sample2train,t_glob,t_limit):

        torch.manual_seed(self.seed*id)
        np.random.seed(self.seed*id)

        self.env = make_env(self.config)
        max_return = self.env.get_max_return()

        o, ep_ret, ep_len, ep_success = self.reset_env(sample2train), 0, 0, 0

        # Main loop: collect experience in env and update/log each epoch
        #env_error_num = 0

        #pbar = tqdm(total = total_steps, desc =str(id) + ". sampler: ", colour="green")
        ep_transitions = []
        t = 0
        while end_flag.value == False:

            while pause_flag.value:
                time.sleep(0.1)
            
            while self.mode_sync == True and t_glob.value >= t_limit.value:
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
                ep_transitions = []
                o, ep_ret, ep_len, ep_success = self.reset_env(sample2train), 0, 0, 0   
                continue     
            
            if bool(info): 
                if 'code' in info: 
                    sample2train.put(info)             
                    if info['code'] < 0:
                        ep_transitions = []
                        o, ep_ret, ep_len, ep_success = self.reset_env(sample2train), 0, 0, 0
                        continue
            
            if r > 0 and r < 100:
                data = {'code': 12, 'description': 'Bonus was used (without task completation)'}
                sample2train.put(data)

            ep_ret += r
            ep_len += 1

            # Ignore the "done" signal if it comes from hitting the time
            # horizon (that is, when it's an artificial terminal signal
            # that isn't based on the agent's state)
            d = False if ep_len==self.max_ep_len else d

            # Store experience to replay buffer
            ep_transitions.append((o, a, r, o2, d))

            if self.heatmap_bool and id == 1:
                self.update_heatmap(a)

            # Super critical, easy to overlook step: make sure to update 
            # most recent observation!
            o = o2

            # End of trajectory handling
            if d or (ep_len == self.max_ep_len):   
                ep_success = ep_ret / float(max_return) if max_return is not None else -1.0
                if self.heatmap_bool and id == 1: 
                    heatmap_pick_send = self.heatmap_pick.copy()
                    heatmap_place_send = self.heatmap_place.copy()
                else:
                    heatmap_pick_send = None
                    heatmap_place_send = None
                data = {'code': 11, 'ep_ret': ep_ret, 'ep_len': ep_len, 'ep_success': ep_success, 'heatmap_pick': heatmap_pick_send, 'heatmap_place': heatmap_place_send}
                sample2train.put(data)

                replay_buffer.store_episode_nstep(ep_transitions,self.n_step,self.gamma)
                
                ep_transitions = []
                o, ep_ret, ep_len = self.reset_env(sample2train), 0, 0               
          
            t += 1           
            #pbar.update(1)   
            #time.sleep(0.1)     

        self.env.shuttdown()        
     