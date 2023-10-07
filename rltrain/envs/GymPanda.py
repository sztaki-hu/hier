import numpy as np
import gymnasium as gym
import panda_gym
import time

from rltrain.envs.Env import Env
from rltrain.envs.builder import make_task

class GymPanda(Env):

    def __init__(self,config,config_framework):
        super(GymPanda, self).__init__(config,config_framework)
        
        # Init Obj Num and goal Num
        if self.task_name in ['PandaReach-v3','PandaReachDense-v3']:
            self.obj_num = 0
            self.goal_num = 1
        elif self.task_name in ['PandaStack-v3','PandaStackDense-v3']:
            self.obj_num = 2
            self.goal_num = 2
        else:
            self.obj_num = 1
            self.goal_num = 1
    
    def reset(self):
        o_dict, _ = self.env.reset()
        o = np.concatenate((o_dict['observation'], o_dict['desired_goal']))
        return o   

    def save_state(self):

        robot_joints = np.array([self.env.robot.get_joint_angle(joint=i) for i in range(7)]) 
        desired_goal = self.env.task.goal
        object_position = self.env.task.get_achieved_goal()

        return robot_joints,desired_goal,object_position

    def get_robot_joints(self):
        return np.array([self.env.robot.get_joint_angle(joint=i) for i in range(7)]) 

    
    def load_state(self,robot_joints,desired_goal,object_position=None):

        if robot_joints is not None: self.env.robot.set_joint_angles(robot_joints)

        if self.goal_num == 1:
            self.env.task.goal = desired_goal
            self.env.task.sim.set_base_pose("target", desired_goal, np.array([0.0, 0.0, 0.0, 1.0]))
        elif self.goal_num == 2:  
            self.env.task.goal = desired_goal
            self.env.task.sim.set_base_pose("target1", desired_goal[:3], np.array([0.0, 0.0, 0.0, 1.0]))
            self.env.task.sim.set_base_pose("target2", desired_goal[3:], np.array([0.0, 0.0, 0.0, 1.0]))  
        else:
            assert False
    
        if self.obj_num == 0:
            pass
        elif self.obj_num == 1:
            self.env.task.sim.set_base_pose("object", object_position, np.array([0.0, 0.0, 0.0, 1.0]))
        elif self.obj_num == 2:
            self.env.task.sim.set_base_pose("object1", object_position[:3], np.array([0.0, 0.0, 0.0, 1.0]))
            self.env.task.sim.set_base_pose("object2", object_position[3:], np.array([0.0, 0.0, 0.0, 1.0]))
        else:
            print("Obj Num: " + str(self.obj_num))
            assert False
      
    # def init_state_valid(self, o):
    #     if self.task_name == 'PandaPush-v3':
    #         o_goal = o[-3:]
    #         o_obj = o[6:9]  
    #         if np.allclose(o_goal, o_obj, rtol=0.0, atol=0.05, equal_nan=False):
    #             return False
        
    #     return True  
    
    
    def step(self,action):

        o_dict, r, terminated, truncated, info = self.env.step(action)

        o = np.concatenate((o_dict['observation'], o_dict['desired_goal']))
       
        if self.reward_shaping_type == 'state_change_bonus':
            r += self.get_reward_bonus(o)
        r = r * self.reward_scalor

        return o, r, terminated, truncated, info 

    def get_reward_bonus(self,o):
        if self.task_name in ['PandaReach-v3','PandaReachDense-v3']:
            return 0 
        if self.task_name in ['PandaPush-v3','PandaPushDense-v3',
                              'PandaSlide-v3','PandaSlideDense-v3']:
            return self.reward_bonus if self.is_diff_state(self.ep_o_start,o) else 0.0
        elif self.task_name in ['PandaPickAndPlace-v3','PandaPickAndPlaceDense-v3']:
            return self.reward_bonus if self.get_achieved_goal_from_obs(o)[2] > 0.25 else 0.0 
        elif self.task_name in ['PandaStack-v3','PandaStackDense-v3']:
            return self.reward_bonus if self.get_achieved_goal_from_obs(o)[5] > 0.25 else 0.0 
        else:
            print(self.task_name)
            assert False
        
    
    # Curriculum Learning ##############################

    def get_init_ranges(self):
        dict = {}
        if self.task_name in ['PandaReach-v3','PandaReachDense-v3']:
            dict['obj_range_low'] = None
            dict['obj_range_high'] = None
            dict['object_size'] = None
        else:
            dict['obj_range_low'] = self.env.task.obj_range_low
            dict['obj_range_high'] = self.env.task.obj_range_high
            dict['object_size'] = self.env.task.object_size
        dict['goal_range_low'] = self.env.task.goal_range_low
        dict['goal_range_high'] = self.env.task.goal_range_high

        dict['obj_num'] = self.obj_num
        dict['goal_num'] = self.goal_num
        
        return dict
    
    def get_obs(self):  
        robot_obs = self.env.robot.get_obs().astype(np.float32)  # robot state
        task_obs = self.env.task.get_obs().astype(np.float32)  # object position, velococity, etc...
        observation = np.concatenate([robot_obs, task_obs])
        o = np.concatenate((observation, self.env.task.get_goal().astype(np.float32)))
        return o
    
    def is_diff_state(self, o_start, o_end, dim = 2, threshold = 0.01):

        obj_start_pos = self.get_achieved_goal_from_obs(o_start)
        obj_end_pos = self.get_achieved_goal_from_obs(o_end)

        distance =  np.linalg.norm(obj_start_pos[:dim] - obj_end_pos[:dim], axis=-1)
    
        return bool(np.array(distance > threshold, dtype=np.float32))

    # def get_first_stable_state_index(self):
    #     if self.task_name in ['PandaReach-v3','PandaReachDense-v3']:
    #         return 0
    #     elif self.task_name in ['PandaPush-v3','PandaPushDense-v3']:
    #         return 0
    #     elif self.task_name in ['PandaSlide-v3','PandaSlideDense-v3']:
    #         return 5
        
    
    # HER ##############################################
    
    def get_achieved_goal_from_obs(self, o):
        if self.task_name in ['PandaReach-v3','PandaReachDense-v3']:
            o2 = o.copy()
            return o2[:3]
        elif self.task_name in ['PandaPush-v3','PandaPushDense-v3']:
            o2 = o.copy()
            return o2[6:9]
        elif self.task_name in ['PandaSlide-v3','PandaSlideDense-v3']:
            o2 = o.copy()
            return o2[6:9]
        elif self.task_name in ['PandaPickAndPlace-v3','PandaPickAndPlaceDense-v3']:
            o2 = o.copy()
            return o2[7:10]
        elif self.task_name in ['PandaStack-v3','PandaStackDense-v3']:
            o2 = o.copy()
            return np.concatenate([o2[7:10],o2[19:22]])

    def get_desired_goal_from_obs(self,o):
        if self.task_name in ['PandaStack-v3','PandaStackDense-v3']:
            return o[-6:].copy()
        else:
            return o[-3:].copy()

    def change_goal_in_obs(self, o, goal):
        o2 = o.copy()
        if self.task_name in ['PandaStack-v3','PandaStackDense-v3']:
            o2[-6:] = goal.copy()
            return o2
        else:
            o2[-3:] = goal.copy()
            return o2
    
    def her_get_reward_and_done(self,o):
        desired_goal = self.get_desired_goal_from_obs(o)
        achieved_goal = self.get_achieved_goal_from_obs(o)

        r = self.env.task.compute_reward(achieved_goal, desired_goal, {})
        d = 1.0 if self.env.task.is_success(achieved_goal, desired_goal) else 0.0
        return r,d
    



            

        
    
    
  
                 

