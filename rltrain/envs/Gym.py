import numpy as np
import gymnasium as gym
import os

from rltrain.envs.Env import Env
from rltrain.envs.builder import make_task

class Gym(Env):
   
    def is_success(self):
        if self.task_name == 'InvertedPendulum-v4':
            return True if self.t == self.max_ep_len else False
        elif self.task_name == 'InvertedDoublePendulum-v4':
            return True if self.t == self.max_ep_len else False

        return False
      
    def step(self,action):

        o, r, terminated, truncated, info = self.env.step(action)
        self.t += 1

        if self.reward_shaping_type == 'sparse':
            r = r * self.reward_scalor
        elif self.reward_shaping_type == 'energy':
            if self.task_name == "MountainCarContinuous-v0":
                energy = 9.81 * abs(o[0]) + 0.5 * o[1]**2 # assuming y is close to abs(x=o[0])
                goal_bonus = self.reward_bonus if r > 0 else 0
                r = r * self.reward_scalor + energy + goal_bonus
            else:
                assert False
        
        info['is_success'] = True if self.is_success() == True else False

        return o, r, terminated, truncated, info

   

    

    
  
                 

