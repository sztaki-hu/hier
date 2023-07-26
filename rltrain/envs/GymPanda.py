import numpy as np
import gymnasium as gym
import panda_gym
import time

from rltrain.envs.Env import Env

class GymPanda(Env):
    def __init__(self,config,config_framework):
        
        self.config = config
        self.config_framework = config_framework
        
        # General
        self.action_space = config['agent']['action_space']
        self.task_name = self.config['environment']['task']['name']
        self.max_ep_len = config['sampler']['max_ep_len'] 

        # Reward
        self.reward_shaping_type = config['environment']['reward']['reward_shaping_type']
        self.reward_scalor = config['environment']['reward']['reward_scalor']
        self.reward_bonus = config['environment']['reward']['reward_bonus']

        # Check validity
        assert self.task_name in self.config_framework['task_list']['gympanda']

        self.npclose = 0
        self.close = 0

        # Create env
        if self.config['environment']['headless']:
            self.env = gym.make(self.task_name)
        else:
            self.env = gym.make(self.task_name, render_mode="human")
        self.env._max_episode_steps = self.max_ep_len

        self.reset()
    
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
    
    
  
                 

