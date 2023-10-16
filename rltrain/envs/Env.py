import time
import numpy as np

from rltrain.envs.builder import make_task

REWARD_TYPES = ['sparse','state_change_bonus']

class Env:

    def __init__(self,config,config_framework):
        
        self.config = config
        self.config_framework = config_framework

        # General
        self.task_name = self.config['environment']['task']['name']
        self.max_ep_len = config['sampler']['max_ep_len']
        self.t = 0         
        self.obs = None  

        # Reward
        self.reward_shaping_type = config['environment']['reward']['reward_shaping_type']
        self.reward_scalor = config['environment']['reward']['reward_scalor']
        self.reward_bonus = config['environment']['reward']['reward_bonus']

        # Create taskenv
        self.env = make_task(config,config_framework)
        self.env._max_episode_steps = int(float(self.max_ep_len))

        assert self.reward_shaping_type in REWARD_TYPES

        self.reset() 
    
    def reset(self):
        o, _ = self.env.reset()
        self.t = 0
        self.obs = o.copy()
        return o       
    
    def save_state(self):
        return None,None,None

    def load_state(self,robot_joints,desired_goal,object_position=None):
        return     
    
    def shuttdown(self):
        self.reset()
        self.env.close()
    
    def init_state_valid(self, o):
        return True  

    def is_success(self):
        return False       

    def step(self,action):

        o, r, terminated, truncated, info = self.env.step(action)

        info['is_success'] = True if self.is_success() == True else False

        self.obs = o.copy()

        return o, r, terminated, truncated, info

    def get_obs(self):
        return self.obs
        
    
    def random_sample(self):
        return self.env.action_space.sample() 

    def get_max_return(self):
        return None
    
    def get_achieved_goal_from_obs(self, o):
        return None
    
    def get_desired_goal_from_obs(self,o):
        return None

    def change_goal_in_obs(self, o, goal):
        return None
    
    def her_get_reward_and_done(self,o):
        return None, None
    
    def get_init_ranges(self):
        return None
    
    def is_diff_state(self,o1,o2,threshold = 0.01):
        return None

    
  
                 

