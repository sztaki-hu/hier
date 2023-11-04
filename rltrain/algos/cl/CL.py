import numpy as np
import random

from typing import Dict, List, Tuple, Union

from rltrain.taskenvs.Gym import Gym
from rltrain.taskenvs.GymPanda import GymPanda

RANGE_GROWTH_MODES = ['simple', 'discard', 'balancediscard']

class CL:
    def __init__(self, config: Dict, taskenv: Union[Gym, GymPanda]):

        # INIT CONFIG
        self.config = config
        self.taskenv = taskenv

        # TASK
        self.task_name = self.config['environment']['task']['name']

        # INIT
        self.total_timesteps = float(config['trainer']['total_timesteps'])
        self.init_ranges = self.taskenv.get_init_ranges()

        print(self.init_ranges)

        if config['trainer']['cl']['type'] != 'nocl':
            self.obj_range_low = self.init_ranges['obj_range_low']
            self.obj_range_high = self.init_ranges['obj_range_high']
            self.object_size = self.init_ranges['object_size']
            self.goal_range_low = self.init_ranges['goal_range_low']
            self.goal_range_high = self.init_ranges['goal_range_high']
            self.obj_num = self.init_ranges['obj_num'] 
            self.goal_num = self.init_ranges['goal_num']

            ###########################################################################
            #   centers: [0, 0, object_size / 2]
            ###########################################################################
            if self.task_name in ['PandaReach-v3','PandaReachDense-v3']:
                ###########################################################################
                # PandaReach-v3:
                #   goal_range:  [-0.15, -0.15, 0] --- [0.15, 0.15, 0.3]
                ###########################################################################
                #self.obj_range_center = (self.obj_range_low + self.obj_range_high) / 2.0
                self.goal_range_center = (self.goal_range_low  + self.goal_range_high) / 2.0

                
            elif self.task_name in ['PandaPush-v3','PandaPushDense-v3']:
                ###########################################################################
                # PandaPush-v3:
                #   goal_range:  [-0.15, -0.15, 0] --- [0.15, 0.15, 0]
                #   obj_range:   [-0.15, -0.15, 0] --- [0.15, 0.15, 0]
                ###########################################################################
                self.obj_range_center = (self.obj_range_low + self.obj_range_high) / 2.0
                self.goal_range_center = (self.goal_range_low  + self.goal_range_high) / 2.0
                # self.obj_range_center[1] += 0.05
                # self.goal_range_center[1] -= 0.05
                self.obj_range_center[2] = self.object_size / 2.0
                self.goal_range_center[2] = self.object_size / 2.0
            elif self.task_name in ['PandaSlide-v3','PandaSlideDense-v3']:
                ###########################################################################
                # PandaSlide-v3:
                #   goal_range:  [-0.35, -0.15, 0] --- [0.35, 0.15, 0]
                #   obj_range:   [-0.15, -0.15, 0] --- [0.15, 0.15, 0]
                ###########################################################################
                self.obj_range_center = (self.obj_range_low + self.obj_range_high) / 2.0
                self.goal_range_center = (self.goal_range_low  + self.goal_range_high) / 2.0
                self.obj_range_center[2] = self.object_size / 2.0
                self.goal_range_center[2] = self.object_size / 2.0
            elif self.task_name in ['PandaPickAndPlace-v3','PandaPickAndPlaceDense-v3']:
                ###########################################################################
                # PandaPickAndPlace-v3:
                #   goal_range:  [-0.15, -0.15, 0] --- [0.15, 0.15, 0.2]
                #   obj_range:   [-0.15, -0.15, 0] --- [0.15, 0.15, 0]
                ###########################################################################
                self.obj_range_center = (self.obj_range_low + self.obj_range_high) / 2.0
                self.goal_range_center = (self.goal_range_low  + self.goal_range_high) / 2.0
                self.obj_range_center[2] = self.object_size / 2.0
                self.goal_range_center[2] = self.object_size / 2.0
            elif self.task_name in ['PandaStack-v3','PandaStackDense-v3']:
                ###########################################################################
                # PandaStack-v3:
                #   goal_range:  [-0.15, -0.15, 0] --- [0.15, 0.15, 0]
                #   obj_range:   [-0.15, -0.15, 0] --- [0.15, 0.15, 0]
                #   centers goal 2: [0, 0, 3 * object_size / 2]
                ###########################################################################
                obj1_range_center = (self.obj_range_low + self.obj_range_high) / 2.0
                goal1_range_center = (self.goal_range_low  + self.goal_range_high) / 2.0     
                obj1_range_center[2] = self.object_size / 2.0
                goal1_range_center[2] = 3* self.object_size / 2.0

                obj2_range_center = (self.obj_range_low + self.obj_range_high) / 2.0
                goal2_range_center = (self.goal_range_low  + self.goal_range_high) / 2.0     
                obj2_range_center[2] = self.object_size / 2.0
                goal2_range_center[2] = 3* self.object_size / 2.0

                self.obj_range_center = np.concatenate([obj1_range_center,obj2_range_center])
                self.goal_range_center = np.concatenate([goal1_range_center,goal2_range_center])
            
            if self.obj_num == 0 and self.goal_num == 1:
                self.obj_range_half = None
                self.goal_range_half = (self.goal_range_high - self.goal_range_low) / 2.0    
            elif self.obj_num == 1 and self.goal_num == 1:     
                self.obj_range_half = (self.obj_range_high - self.obj_range_low) / 2.0
                self.goal_range_half = (self.goal_range_high - self.goal_range_low) / 2.0 
            elif self.obj_num == 2 and self.goal_num == 2:
                obj1_range_half = (self.obj_range_high - self.obj_range_low) / 2.0
                goal1_range_half = (self.goal_range_high - self.goal_range_low) / 2.0
                
                obj2_range_half = (self.obj_range_high - self.obj_range_low) / 2.0
                goal2_range_half = (self.goal_range_high - self.goal_range_low) / 2.0

                self.obj_range_half = np.concatenate([obj1_range_half,obj2_range_half])
                self.goal_range_half = np.concatenate([goal1_range_half,goal2_range_half])
            else:
                print("Obj Num: " + str(self.obj_num))
                print("Goal Num: " + str(self.goal_num))
                assert False
                
            self.cl_range_growth_mode = config['trainer']['cl']['range_growth_mode']
            self.balancediscard_ratio = config['trainer']['cl']['balancediscard_ratio']
            self.cl_ratio_discard_lag = self.config['trainer']['cl']['ratio_discard_lag']

            assert self.cl_range_growth_mode in RANGE_GROWTH_MODES

        self.cl_ratio = 0
        self.cl_obj_ratio = 0
        self.cl_goal_ratio = 0

        self.cl_ratio_discard = 0
        self.cl_obj_ratio_discard = 0
        self.cl_goal_ratio_discard = 0

        self.store_rollout_success_rate = False
        self.store_eval_success_rate = False
    
    def reset_env(self,t):

        self.update_cl(t)

        desired_goal,object_position = self.get_setup()

        self.taskenv.reset()
        self.taskenv.load_state(robot_joints= None, desired_goal = desired_goal, object_position = object_position)
        
        return self.taskenv.get_obs()

    def update_cl(self,t):
        pass

    def get_setup(self):

        if self.cl_range_growth_mode == "simple": return self.get_range_rectangle() 
        elif self.cl_range_growth_mode == "discard": return self.get_range_rectangle_with_cutout()
        elif self.cl_range_growth_mode == "balancediscard": return self.get_range_rectangle_with_cutout() if random.random() < 0.80 else self.get_range_rectangle()


    def get_range_rectangle(self):

        goal_low, goal_high, obj_low, obj_high = self.get_range(self.cl_obj_ratio, self.cl_goal_ratio)
        
        if self.obj_num == 0:
            object_position = None
            desired_goal = np.random.uniform(goal_low, goal_high)
        else:
            object_position =  np.random.uniform(obj_low, obj_high)
            desired_goal = np.random.uniform(goal_low, goal_high)
            

        return desired_goal,object_position
    
    def get_range_rectangle_with_cutout(self):

        goal_low, goal_high, obj_low, obj_high = self.get_range(self.cl_obj_ratio, self.cl_goal_ratio)
        goal_d_low, goal_d_high, obj_d_low, obj_d_high = self.get_range(self.cl_obj_ratio_discard, self.cl_goal_ratio_discard)
        
        if random.random() < 0.50:
            desired_goal = np.random.uniform(goal_low, goal_d_low)
        else:
            desired_goal = np.random.uniform(goal_d_high, goal_high)
        
        if self.obj_num == 0:
            object_position = None
        else:
            if random.random() < 0.50:
                object_position =  np.random.uniform(obj_low, obj_d_low)
            else:
                object_position =  np.random.uniform(obj_d_high, obj_high)
        
        return desired_goal,object_position
    
    def get_range(self,obj_ratio,goal_ratio):     
        if self.obj_num == 0:
            obj_low = None
            obj_high = None
        else:
            obj_low = self.obj_range_center - self.obj_range_half * obj_ratio
            obj_high = self.obj_range_center + self.obj_range_half * obj_ratio
        goal_low = self.goal_range_center - self.goal_range_half * goal_ratio
        goal_high = self.goal_range_center + self.goal_range_half * goal_ratio

        return goal_low, goal_high, obj_low, obj_high
    
    def copy_cl_ratios_to_obj_and_goal(self):
        self.cl_obj_ratio = self.cl_ratio
        self.cl_goal_ratio = self.cl_ratio
        self.cl_obj_ratio_discard = self.cl_ratio_discard
        self.cl_goal_ratio_discard = self.cl_ratio_discard
    



    




     