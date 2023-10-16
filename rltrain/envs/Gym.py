import numpy as np
import gymnasium as gym
import os

from rltrain.envs.Env import Env
from rltrain.envs.builder import make_task

class Gym(Env):

    def __init__(self,config,config_framework):
        super(Gym, self).__init__(config,config_framework)

        assert config['buffer']['her']['goal_selection_strategy'] == 'noher'
        assert config['trainer']['cl']['type'] == 'nocl'
   
    def step(self,action):

        o, r, terminated, truncated, info = self.env.step(action)
        
        self.t += 1
        info['is_success'] = True if self.t == self.max_ep_len else False
        
        r = r * self.reward_scalor 

        self.obs = o.copy()

        return o, r, terminated, truncated, info
    


   

    

    
  
                 

