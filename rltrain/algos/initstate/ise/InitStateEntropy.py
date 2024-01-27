import numpy as np
import random
import collections

from typing import Dict, List, Tuple, Union, Optional
from rltrain.taskenvs.GymPanda import GymPanda
from rltrain.algos.initstate.InitState import InitState

RANGE_GROWTH_MODES = ['simple', 'discard', 'balancediscard']

class InitStateEntropy(InitState):
    def __init__(self, config: Dict, taskenv: GymPanda) -> None:

        # INIT CONFIG
        self.config = config
        self.taskenv = taskenv

        # TASK
        self.task_name = self.config['environment']['task']['name']

        # INIT
        self.total_timesteps = float(config['trainer']['total_timesteps'])
        self.init_ranges = self.taskenv.get_init_ranges()

        print(self.init_ranges)

        if config['trainer']['init_state']['type'] not in ['min','max']:
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
                raise ValueError("[ISE]: Obj num and/or goal num are not valid. Obj num = " + str(self.obj_num) + " | Goal num = " + str(self.goal_num))
                
            self.range_growth_mode = config['trainer']['init_state']['ise']['range_growth_mode']
            self.balancediscard_ratio = config['trainer']['init_state']['ise']['balancediscard_ratio']
            self.c_discard_lag = self.config['trainer']['init_state']['ise']['ratio_discard_lag']

            assert self.range_growth_mode in RANGE_GROWTH_MODES

        self.c = 0
        self.c_obj = 0
        self.c_goal = 0

        self.c_discard = 0
        self.c_obj_discard = 0
        self.c_goal_discard = 0

        self.store_rollout_success_rate = False
        self.store_eval_success_rate = False

        self.eval_success_dq = collections.deque()
        self.rollout_success_dq = collections.deque() 

        # goal_low, goal_high, obj_low, obj_high = self.get_range(self.c_obj, self.c_goal)

        # print(goal_low)
        # print(type(goal_low))
        # print(obj_low)
        # print(type(obj_low))

        # assert False
    
    def reset_env(self, t: int) -> np.ndarray:

        self.update_c(t)

        desired_goal,object_position = self.get_setup()

        self.taskenv.reset()
        self.taskenv.load_state(robot_joints= None, desired_goal = desired_goal, object_position = object_position)
        
        return self.taskenv.get_obs()

    def update_c(self,t: int) -> None:
        pass

    def get_setup(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:

        if self.range_growth_mode == "simple": return self.get_range_rectangle() 
        elif self.range_growth_mode == "discard": return self.get_range_rectangle_with_cutout()
        elif self.range_growth_mode == "balancediscard": return self.get_range_rectangle_with_cutout() if random.random() < 0.80 else self.get_range_rectangle()
        else: raise ValueError("[ISE]: range_growth_mode: '" + str(self.range_growth_mode) + "' must be in : " + str(RANGE_GROWTH_MODES))
   


    def get_range_rectangle(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:

        goal_low, goal_high, obj_low, obj_high = self.get_range(self.c_obj, self.c_goal)
        
        if self.obj_num == 0:
            object_position = None
            desired_goal = np.random.uniform(goal_low, goal_high)
        else:
            if obj_low is not None and obj_high is not None:
                object_position =  np.random.uniform(obj_low, obj_high)
                desired_goal = np.random.uniform(goal_low, goal_high)
            else:
                raise ValueError("[ISE]: [obj_low, obj_high]: " + str([obj_low, obj_high]))
            

        return desired_goal,object_position
    
    def get_range_rectangle_with_cutout(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:

        goal_low, goal_high, obj_low, obj_high = self.get_range(self.c_obj, self.c_goal)
        goal_d_low, goal_d_high, obj_d_low, obj_d_high = self.get_range(self.c_obj_discard, self.c_goal_discard)
        
        if random.random() < 0.50:
            desired_goal = np.random.uniform(goal_low, goal_d_low)
        else:
            desired_goal = np.random.uniform(goal_d_high, goal_high)
        
        if self.obj_num == 0:
            object_position = None
        else:        
            if random.random() < 0.50:
                if obj_low is not None and obj_d_low is not None:
                    object_position =  np.random.uniform(obj_low, obj_d_low)
                else:
                    raise ValueError("[ISE]: [obj_low, obj_d_low]: " + str([obj_low, obj_d_low]))
            else:
                if obj_d_high is not None and obj_high is not None:
                    object_position =  np.random.uniform(obj_d_high, obj_high)
                else:
                    raise ValueError("[ISE]: [obj_d_high, obj_high]: " + str([obj_d_high, obj_high]))
        
        return desired_goal,object_position
    
    def get_range(self, 
                  obj_ratio: float, 
                  goal_ratio: float
                  ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:    
        if self.obj_num == 0:
            obj_low = None
            obj_high = None
        else:
            if self.obj_range_half is not None:
                obj_low = self.obj_range_center - self.obj_range_half * obj_ratio
                obj_high = self.obj_range_center + self.obj_range_half * obj_ratio
            else:
                raise ValueError("[ISE]: obj_range_half: " + str(self.obj_range_half))
        goal_low = self.goal_range_center - self.goal_range_half * goal_ratio
        goal_high = self.goal_range_center + self.goal_range_half * goal_ratio

        return goal_low, goal_high, obj_low, obj_high
    
    
    def append_rollout_success_dq(self, ep_succes: float) -> None:
        if self.store_rollout_success_rate: self.rollout_success_dq.append(ep_succes)
    
    def append_eval_success_dq(self, ep_succes: float) -> None:
        if self.store_rollout_success_rate: self.eval_success_dq.append(ep_succes)
    
    def copy_c_to_obj_and_goal(self):
        self.c_obj = self.c
        self.c_goal = self.c
        self.c_obj_discard = self.c_discard
        self.c_goal_discard = self.c_discard
    



    




     