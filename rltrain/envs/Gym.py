import numpy as np
import gymnasium as gym
import time

from rltrain.envs.Env import Env

REWARD_TYPE_LIST = ['sparse','energy']

class Gym(Env):
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

        # Check validity       
        assert self.task_name in self.config_framework['task_list']['gym']
        assert self.reward_shaping_type in REWARD_TYPE_LIST

        # Create env
        if self.config['environment']['headless']:
            self.env = gym.make(self.task_name)
        else:
            self.env = gym.make(self.task_name, render_mode="human")
        self.env._max_episode_steps = self.max_ep_len

        self.reset() 
    
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

   

    

    
  
                 

