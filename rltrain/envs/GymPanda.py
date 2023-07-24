import numpy as np
import gymnasium as gym
import panda_gym
import time


REWARD_TYPE_LIST = ['sparse']
TASK_LIST = ['PandaReach-v3','PandaPush-v3']

class GymPanda:
    def __init__(self,config):
        self.config = config

        self.reward_shaping_type = config['environment']['reward']['reward_shaping_type']
        self.reward_scalor = config['environment']['reward']['reward_scalor']
        self.reward_bonus = config['environment']['reward']['reward_bonus']

        self.obs_dim = config['environment']['obs_dim']
        self.action_space = config['agent']['action_space']
        self.task_name = self.config['environment']['task']['name']
        self.task_params = self.config['environment']['task']['params']
        self.act_dim = config['environment']['act_dim']
        self.boundary_min = np.array(config['agent']['boundary_min'])[:self.act_dim]
        self.boundary_max = np.array(config['agent']['boundary_max'])[:self.act_dim]

        self.max_ep_len = config['sampler']['max_ep_len'] 

        assert self.reward_shaping_type in REWARD_TYPE_LIST
        assert self.task_name in TASK_LIST

        if self.config['environment']['headless']:
            self.env = gym.make(self.task_name)
        else:
            self.env = gym.make(self.task_name, render_mode="human")
        self.env._max_episode_steps = self.max_ep_len

        self.reset()

    
    def shuttdown(self):
        self.reset()
        self.env.close()
    
    def reset_once(self):
        return self.reset()

    def reset(self):
        o_dict, info = self.env.reset()
        o = np.concatenate((o_dict['observation'], o_dict['desired_goal']))
        return o           

    def init_state_valid(self):
        return True  
    
    def step(self,action):

        o_dict, r, terminated, truncated, info = self.env.step(action)

        o = np.concatenate((o_dict['observation'], o_dict['desired_goal']))
       
        r = r * self.reward_scalor


        return o, r, terminated, truncated, info
    
    def random_sample(self):
        return self.env.action_space.sample()
    
    def get_max_return(self):
        return None
    
  
                 

