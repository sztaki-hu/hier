import numpy as np
import gymnasium as gym
import panda_gym
import time



TASK_LIST = ['PandaReach-v3','PandaPush-v3','PandaSlide-v3']

class GymPanda:
    def __init__(self,config):
        self.config = config

        # General
        self.action_space = config['agent']['action_space']
        self.task_name = self.config['environment']['task']['name']
        self.max_ep_len = config['sampler']['max_ep_len'] 

        # Reward
        self.reward_shaping_type = config['environment']['reward']['reward_shaping_type']
        self.reward_scalor = config['environment']['reward']['reward_scalor']
        self.reward_bonus = config['environment']['reward']['reward_bonus']

        # Check validity
        assert self.task_name in TASK_LIST

        self.npclose = 0
        self.close = 0

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
        o_dict, info = self.env.reset()
        o = np.concatenate((o_dict['observation'], o_dict['desired_goal']))
        return o     

    def reset_with_init_check(self):
        init_invalid_num = 0
        reset_num = 0
        ## Reset Env
        while True:
            try:
                o = self.reset()
                reset_num += 1
                if self.init_state_valid(o):
                    info = {}
                    info['init_invalid_num'] = init_invalid_num
                    info['reset_num'] = reset_num
                    return o, info
                else:
                    init_invalid_num+=0                   
            except:        
                time.sleep(0.1)      

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
    
    def random_sample(self):
        return self.env.action_space.sample()
    
    def get_max_return(self):
        return None
    
  
                 

