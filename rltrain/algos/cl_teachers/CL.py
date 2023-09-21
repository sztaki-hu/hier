import numpy as np

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

        if self.task_name == 'PandaPush-v3':
            self.goal_range_center[1] -= 0.05
            self.obj_range_center[1] += 0.05
        self.goal_range_center[2] = self.object_size / 2.0
        self.obj_range_center[2] = self.object_size / 2.0

        self.goal_range_half = (self.goal_range_high - self.goal_range_low) / 2.0
        self.obj_range_half = (self.obj_range_high - self.obj_range_low) / 2.0
    
    def get_range(self):        
        goal_low = self.goal_range_center - self.goal_range_half * self.t_ratio
        goal_high = self.goal_range_center + self.goal_range_half * self.t_ratio
        obj_low = self.obj_range_center - self.obj_range_half * self.t_ratio
        obj_high = self.obj_range_center + self.obj_range_half * self.t_ratio
        return goal_low, goal_high, obj_low, obj_high

    def reset_env(self,t):

        self.update_ratio(t)
        goal_low, goal_high, obj_low, obj_high = self.get_range()
        
        desired_goal = np.random.uniform(goal_low, goal_high)
        object_position =  np.random.uniform(obj_low, obj_high)

        self.env.reset()
        self.env.load_state(robot_joints= None, desired_goal = desired_goal, object_position = object_position)
        
        return self.env.get_obs()

    def update_ratio(self,t):
        pass
    




     