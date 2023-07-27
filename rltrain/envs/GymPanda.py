import numpy as np
import gymnasium as gym
import panda_gym
import time

from rltrain.envs.Env import Env
from rltrain.envs.builder import make_task

class GymPanda(Env):
    
    def reset(self):
        o_dict, _ = self.env.reset()
        o = np.concatenate((o_dict['observation'], o_dict['desired_goal']))
        return o    


    def init_state_valid(self, o):
        if self.task_name == 'PandaPush-v3':
            o_goal = o[-3:]
            o_obj = o[6:9]  
            if np.allclose(o_goal, o_obj, rtol=0.0, atol=0.05, equal_nan=False):
                return False
        
        return True  
    
    def step(self,action):

        o_dict, r, terminated, truncated, info = self.env.step(action)

        o = np.concatenate((o_dict['observation'], o_dict['desired_goal']))
       
        r = r * self.reward_scalor


        return o, r, terminated, truncated, info
    
    
  
                 

