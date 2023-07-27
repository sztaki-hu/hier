import numpy as np
import gymnasium as gym
import time

from rltrain.envs.builder import make_task

class Env:

    def __init__(self,config,config_framework):
        
        self.config = config
        self.config_framework = config_framework

        # General
        self.task_name = self.config['environment']['task']['name']
        self.action_space = config['agent']['action_space']
        self.max_ep_len = config['sampler']['max_ep_len']
        self.t = 0           

        # Reward
        self.reward_shaping_type = config['environment']['reward']['reward_shaping_type']
        self.reward_scalor = config['environment']['reward']['reward_scalor']
        self.reward_bonus = config['environment']['reward']['reward_bonus']

        # Create taskenv
        self.env = make_task(config,config_framework)
        self.env._max_episode_steps = self.max_ep_len

        self.reset() 
    
    def reset(self):
        o, _ = self.env.reset()
        return o       
    
    def shuttdown(self):
        self.reset()
        self.env.close()
    
    def init_state_valid(self, o):
        return True  

    def reset_with_init_check(self):
        init_invalid_num = 0
        reset_num = 0
        ## Reset Env
        while True:
            o = self.reset()
            try:
                o = self.reset()
                reset_num += 1
                if self.init_state_valid(o):
                    info = {}
                    info['init_invalid_num'] = init_invalid_num
                    info['reset_num'] = reset_num
                    return o, info
                else:
                    init_invalid_num+=0                
            except:        
                time.sleep(0.1) 

    def is_success(self):
        return False       

    def step(self,action):

        o, r, terminated, truncated, info = self.env.step(action)

        info['is_success'] = True if self.is_success() == True else False

        return o, r, terminated, truncated, info
    
    def random_sample(self):
        return self.env.action_space.sample() 

    def get_max_return(self):
        return None
    

    
  
                 

