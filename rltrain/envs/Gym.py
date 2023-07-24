import numpy as np
import gymnasium as gym
import time


REWARD_TYPE_LIST = ['sparse','energy']
TASK_LIST = ['MountainCarContinuous-v0','InvertedPendulum-v4','InvertedDoublePendulum-v4','Swimmer-v4',
             'Hopper-v4','HalfCheetah-v4','Walker2d-v4','Ant-v4','Reacher-v4','Humanoid-v4','HumanoidStandup-v4','Pusher-v4']

class Gym:
    def __init__(self,config):
        
        self.config = config

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
        assert self.reward_shaping_type in REWARD_TYPE_LIST
        assert self.task_name in TASK_LIST

        # Create env
        if self.config['environment']['headless']:
            self.env = gym.make(self.task_name)
        else:
            self.env = gym.make(self.task_name, render_mode="human")
        self.env._max_episode_steps = self.max_ep_len

        self.reset()

    
    def shuttdown(self):
        self.reset()
        self.env.close()

    def reset(self):
        o, info = self.env.reset()
        return o       

    def reset_with_init_check(self):
        ## Reset Env
        while True:
            try:
                o = self.reset()
                if self.init_state_valid(o):
                    return o
                else:                   
                    time.sleep(0.1)
            except:        
                time.sleep(1)       

    def init_state_valid(self):
        return True  
    
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
    
    def random_sample(self):
        return self.env.action_space.sample()

    def is_success(self):
        if self.task_name == 'InvertedPendulum-v4':
            return True if self.t == self.max_ep_len else False
        elif self.task_name == 'InvertedDoublePendulum-v4':
            return True if self.t == self.max_ep_len else False

        return False

    def get_max_return(self):
        return None
    

    
  
                 

