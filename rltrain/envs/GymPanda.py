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
    
    def get_goal_state_from_obs(self, o):
        if self.task_name == 'PandaReach-v3':
            o2 = o.copy()
            return o2[:3]
        elif self.task_name == 'PandaPush-v3':
            o2 = o.copy()
            return o2[6:9]
        elif self.task_name == 'PandaSlide-v3':
            o2 = o.copy()
            return o2[6:9]

    def change_goal_in_obs(self, o, goal):
        o2 = o.copy()
        o2[-3:] = goal.copy()
        return o2
    
    def her_get_reward_and_done(self,o):
        o_goal = o[-3:]
        if self.task_name == 'PandaReach-v3': 
            o_cond = o[:3] 
        elif self.task_name == 'PandaPush-v3':
            o_cond = o[6:9] 
        elif self.task_name == 'PandaSlide-v3':
            o_cond = o[6:9] 
            
        if np.allclose(o_goal, o_cond, rtol=0.0, atol=0.01, equal_nan=False):
            return 0,1
        else:
            return -1,0
        
    
    
  
                 

