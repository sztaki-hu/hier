import os
import math
import numpy as np
import torch
import time

from tqdm import tqdm

from rltrain.envs.builder import make_env

class Eval:

    def __init__(self,agent,logger,config,config_framework):
        #self.env = env
        self.agent = agent
        self.logger = logger
        self.config = config
        self.config_framework = config_framework

        self.max_ep_len = config['sampler']['max_ep_len']  
        self.agent_type = config['agent']['type']  

    
    def eval_agent(self,model_name=90,num_display_episode=10, headless=True, time_delay=0.02):

        # Load model
        path = self.logger.get_model_save_path(model_name)

        self.agent.load_weights(path,mode="pi",eval=True)

        # Create Env
        self.config['environment']['headless'] = headless
        self.env = make_env(self.config,self.config_framework)

        # Start Eval
        ep_rets = []
        ep_lens = []
        success_num = 0.0
        for j in range(num_display_episode):
            [o, info], d, ep_ret, ep_len = self.env.reset_with_init_check(), False, 0, 0
            goal = self.env.get_desired_goal_from_obs(o)
            while not(d or (ep_len == self.max_ep_len)):
                # Take deterministic actions at test time 
                if self.agent_type == 'sac':
                    a = self.agent.get_action(o, True)
                elif self.agent_type == 'td3':
                    a = self.agent.get_action(o, 0)
                o, r, terminated, truncated, info = self.env.step(a)
                d = terminated or truncated
                ep_ret += r
                ep_len += 1
                if headless == False: time.sleep(time_delay)
            ep_rets.append(ep_ret)
            ep_lens.append(ep_len)
            if info['is_success'] == True: success_num += 1
            print("--------------------------------")
            print("ep_ret: " + str(ep_ret) + " | ep_len : " + str(ep_len) + " | Success: " + str(info['is_success']) + " | Goal: " + str(goal))
        
        ep_ret_avg = sum(ep_rets) / len(ep_rets)
        mean_ep_length = sum(ep_lens) / len(ep_lens)
        success_rate = success_num / num_display_episode
        
        print("#########################################")
        print("ep_ret_avg: " + str(ep_ret_avg))
        print("mean_ep_length: " + str(mean_ep_length))
        print("success_rate: " + str(success_rate))
        print("#########################################")
        
        # Shutdown environment
        self.env.shuttdown() 


