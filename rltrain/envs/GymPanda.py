import numpy as np
import gymnasium as gym
import panda_gym
import time

from rltrain.envs.Env import Env
from rltrain.envs.builder import make_task

class GymPanda(Env):
    
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
        if desired_goal is not None:
            self.env.task.goal = desired_goal
            self.env.task.sim.set_base_pose("target", desired_goal, np.array([0.0, 0.0, 0.0, 1.0]))
    
        if self.task_name == 'PandaPush-v3' or self.task_name == 'PandaSlide-v3':
            if object_position is not None: self.env.task.sim.set_base_pose("object", object_position, np.array([0.0, 0.0, 0.0, 1.0]))


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
       
        r = r * self.reward_scalor

        return o, r, terminated, truncated, info 
    
    # Curriculum Learning ##############################

    def get_init_ranges(self):
        dict = {}
        dict['goal_range_low'] = self.env.task.goal_range_low
        dict['goal_range_high'] = self.env.task.goal_range_high
        dict['obj_range_low'] = self.env.task.obj_range_low
        dict['obj_range_high'] = self.env.task.obj_range_high
        dict['object_size'] = self.env.task.object_size
        return dict
    
    def get_obs(self):  
        robot_obs = self.env.robot.get_obs().astype(np.float32)  # robot state
        task_obs = self.env.task.get_obs().astype(np.float32)  # object position, velococity, etc...
        observation = np.concatenate([robot_obs, task_obs])
        o = np.concatenate((observation, self.env.task.get_goal().astype(np.float32)))
        return o
        
    
    # HER ##############################################
    
    def get_achieved_goal_from_obs(self, o):
        if self.task_name == 'PandaReach-v3':
            o2 = o.copy()
            return o2[:3]
        elif self.task_name == 'PandaPush-v3':
            o2 = o.copy()
            return o2[6:9]
        elif self.task_name == 'PandaSlide-v3':
            o2 = o.copy()
            return o2[6:9]

    def get_desired_goal_from_obs(self,o):
        return o[-3:].copy()

    def change_goal_in_obs(self, o, goal):
        o2 = o.copy()
        o2[-3:] = goal.copy()
        return o2
    
    def her_get_reward_and_done(self,o):
        desired_goal = self.get_desired_goal_from_obs(o)
        achieved_goal = self.get_achieved_goal_from_obs(o)

        r = self.env.task.compute_reward(achieved_goal, desired_goal, {})
        d = 1 if r == 0 else 0
        return r,d
    



            

        
    
    
  
                 

