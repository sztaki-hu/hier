import numpy as np

from rlbench.observation_config import ObservationConfig, CameraConfig
from pyrep.const import RenderMode

from rlbench.environment import Environment
from rlbench.action_modes.action_mode import  MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete

from rltrain.envs.Env import Env
from rltrain.envs.builder import make_task

class RLBenchJoint(Env):
    def __init__(self,config,config_framework):

        self.config = config
        self.config_framework = config_framework

        # General
        self.task_name = self.config['environment']['task']['name']
        self.max_ep_len = config['sampler']['max_ep_len']
        self.t = 0     

        # Boundary
        self.act_dim = config['environment']['act_dim']
        self.boundary_min = np.array(config['agent']['boundary_min'])[:self.act_dim]
        self.boundary_max = np.array(config['agent']['boundary_max'])[:self.act_dim]      

        # Reward
        self.reward_shaping_type = config['environment']['reward']['reward_shaping_type']
        self.reward_scalor = config['environment']['reward']['reward_scalor']
        self.reward_bonus = config['environment']['reward']['reward_bonus']

        # Create taskenv

        self.env = make_task(config,config_framework)

        self.env.launch()
        
        self.task_env = self.env.get_task(self.env._string_to_task(self.task_name +'.py'))

        self.reset_with_init_check()

    
    def reset(self):
        self.task_env.reset()
        o = self.get_obs()
        return o   
    
    def step(self,action):

        _, r, d, info,  = self.task_env.step(action)

        o = self.get_obs()

        terminated = d
        truncated = True if self.t == self.max_ep_len else False

        info['is_success'] = True if self.is_success(r) == True else False

        r = r * self.reward_scalor

        return o, r, terminated, truncated, info
    
    def random_sample(self):
        return np.random.uniform(low=self.boundary_min, high=self.boundary_max, size=(self.act_dim))

    def get_obs(self):

        robot_obs = self.task_env._scene.robot.arm.get_joint_positions()
        task_obs = self.task_env._scene.task.get_real_obs()

        obs = np.concatenate([robot_obs, task_obs])

        return obs
    
    def is_success(self,r):
        if self.task_name == "reach_target_no_distractors":
            return True if r == 1 else False 

        return False     
       
