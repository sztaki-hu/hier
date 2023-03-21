import numpy as np
import gym


REWARD_TYPE_LIST = ['sparse','energy']

class Gym:
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

        if self.config['environment']['headless']:
            self.env = gym.make(self.task_name)
        else:
            # self.env = gym.make(self.task_name,render_mode="human") new version
            self.env = gym.make(self.task_name) # old gym version 0.21.0
        self.env._max_episode_steps = self.max_ep_len

        self.reset()

    
    def shuttdown(self):
        self.reset()
        self.env.close()
    
    def reset_once(self):
        return self.reset()

    def reset(self):
        o, info = self.env.reset()
        if self.task_name == "MountainCarContinuous-v0": # for old gym version (0.21.0)
            o = np.array([o,0]) 
        return o           

    def init_state_valid(self):
        return True  
    
    def step(self,action):

        # New gym version
        # o, r, terminated, truncated, info = self.env.step(action)
        # d = terminated or truncated

        # Old gym version (0.21.0)
        o, r, d, info = self.env.step(action)

        if self.reward_shaping_type == 'sparse':
            r = r * self.reward_scalor
        elif self.reward_shaping_type == 'energy':
            energy = 9.81 * abs(o[0]) + 0.5 * o[1]**2 # assuming y is close to abs(x=o[0])
            goal_bonus = self.reward_bonus if r > 0 else 0
            r = r * self.reward_scalor + energy + goal_bonus

        return o, r, d, info
    
    def render(self): # for old gym version 0.21.0
        self.env.render()
    
    def get_max_return(self):
        return None
    
  
                 
