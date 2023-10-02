import numpy as np
import random

RANGE_GROWTH_MODES = ['simple', 'discard', 'balancediscard']

class CL:
    def __init__(self,config, env, replay_buffer):

        # INIT CONFIG
        self.config = config
        self.env = env
        self.replay_buffer = replay_buffer

        # TASK
        self.task_name = self.config['environment']['task']['name']

        # INIT
        self.total_timesteps = float(config['trainer']['total_timesteps'])
        self.init_ranges = self.env.get_init_ranges()

        print(self.init_ranges)
       
        self.goal_range_low = self.init_ranges['goal_range_low']
        self.goal_range_high = self.init_ranges['goal_range_high']
        self.obj_range_low = self.init_ranges['obj_range_low']
        self.obj_range_high = self.init_ranges['obj_range_high']
        self.object_size = self.init_ranges['object_size']

        self.goal_range_low[2] = self.object_size / 2.0
        self.goal_range_high[2] = self.object_size / 2.0
        self.obj_range_low[2] = self.object_size / 2.0
        self.obj_range_high[2] = self.object_size / 2.0

        self.goal_range_center = (self.goal_range_low  + self.goal_range_high) / 2.0
        self.obj_range_center = (self.obj_range_low + self.obj_range_high) / 2.0

        if self.task_name in ['PandaPush-v3','PandaPushDense-v3']:
            self.goal_range_center[1] -= 0.05
            self.obj_range_center[1] += 0.05
        self.goal_range_center[2] = self.object_size / 2.0
        self.obj_range_center[2] = self.object_size / 2.0

        self.goal_range_half = (self.goal_range_high - self.goal_range_low) / 2.0
        self.obj_range_half = (self.obj_range_high - self.obj_range_low) / 2.0

        self.cl_range_growth_mode = config['trainer']['cl']['range_growth_mode']
        self.balancediscard_ratio = config['trainer']['cl']['balancediscard_ratio']
        self.cl_ratio_discard_lag = self.config['trainer']['cl']['ratio_discard_lag']

        assert self.cl_range_growth_mode in RANGE_GROWTH_MODES

        self.cl_ratio = 1
        self.store_success_rate = False
    
    def reset_env(self,t):

        self.update_cl(t)

        desired_goal,object_position = self.get_setup()

        self.env.reset()
        self.env.load_state(robot_joints= None, desired_goal = desired_goal, object_position = object_position)
        
        return self.env.get_obs()

    def update_cl(self,t):
        pass

    def get_setup(self):

        if self.cl_range_growth_mode == "simple": return self.get_range_rectangle() 
        elif self.cl_range_growth_mode == "discard": return self.get_range_rectangle_with_cutout()
        elif self.cl_range_growth_mode == "balancediscard": return self.get_range_rectangle_with_cutout() if random.random() < 0.80 else self.get_range_rectangle()


    def get_range_rectangle(self):

        goal_low, goal_high, obj_low, obj_high = self.get_range(self.cl_ratio)
            
        desired_goal = np.random.uniform(goal_low, goal_high)
        object_position =  np.random.uniform(obj_low, obj_high)

        return desired_goal,object_position
    
    def get_range_rectangle_with_cutout(self):

        goal_low, goal_high, obj_low, obj_high = self.get_range(self.cl_ratio)
        goal_d_low, goal_d_high, obj_d_low, obj_d_high = self.get_range(self.cl_ratio_discard)
        
        if random.random() < 0.50:
            desired_goal = np.random.uniform(goal_low, goal_d_low)
        else:
            desired_goal = np.random.uniform(goal_d_high, goal_high)
        
        if random.random() < 0.50:
            object_position =  np.random.uniform(obj_low, obj_d_low)
        else:
            object_position =  np.random.uniform(obj_d_high, obj_high)
        
        return desired_goal,object_position
    
    def get_range(self,ratio):        
        goal_low = self.goal_range_center - self.goal_range_half * ratio
        goal_high = self.goal_range_center + self.goal_range_half * ratio
        obj_low = self.obj_range_center - self.obj_range_half * ratio
        obj_high = self.obj_range_center + self.obj_range_half * ratio
        return goal_low, goal_high, obj_low, obj_high


    




     