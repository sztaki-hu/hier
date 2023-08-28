import numpy as np
import random

class PredefinedCL:
    def __init__(self,config, env, replay_buffer):
        self.config = config
        self.env = env
        self.replay_buffer = replay_buffer

        self.total_timesteps = config['trainer']['total_timesteps']
        self.init_ranges = self.env.get_init_ranges()

        print(self.init_ranges )
       
        self.goal_range_low = self.init_ranges['goal_range_low']
        self.goal_range_high = self.init_ranges['goal_range_high']
        self.obj_range_low = self.init_ranges['obj_range_low']
        self.obj_range_high = self.init_ranges['obj_range_high']
        self.object_size = self.init_ranges['object_size']

        self.goal_range_low[2] = self.object_size / 2.0
        self.goal_range_high[2] = self.object_size / 2.0
        self.obj_range_low[2] = self.object_size / 2.0
        self.obj_range_high[2] = self.object_size / 2.0

        self.goal_mu = (self.goal_range_low  + self.goal_range_high) / 2.0
        self.obj_mu = (self.obj_range_low + self.obj_range_high) / 2.0

        self.goal_mu[1] -= 0.05
        self.obj_mu[1] += 0.05
        self.goal_mu[2] = self.object_size / 2.0
        self.obj_mu[2] = self.object_size / 2.0

    
    def reset_env(self,t):

        self.env.reset()

        t_ratio = t / float(self.total_timesteps)
        if t_ratio <  0.2:
            self.env.load_state(robot_joints= None, desired_goal = self.goal_mu, object_position = self.obj_mu)
        elif t_ratio < 0.8:
            sigma = np.array([t_ratio/5.0,t_ratio/5.0,0.0])

            desired_goal = np.clip(np.random.normal(self.goal_mu, sigma), self.goal_range_low, self.goal_range_high)
            object_position = np.clip(np.random.normal(self.obj_mu, sigma), self.obj_range_low, self.obj_range_high)

            self.env.load_state(robot_joints= None, desired_goal = desired_goal, object_position = object_position)
        
        return self.env.get_obs()
    




     