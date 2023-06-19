import numpy as np
import time 

from rlbench.observation_config import ObservationConfig, CameraConfig
from pyrep.const import RenderMode

from rlbench.environment import Environment
from rlbench.action_modes.action_mode import ActionMode, MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import ArmActionMode, JointPosition, EndEffectorPoseViaPlanning, EndEffectorPoseViaIK, JointVelocity
from rlbench.action_modes.gripper_action_modes import GripperActionMode, Discrete, Closed

from rlbench.task_environment import TaskEnvironment
from rlbench.backend.observation import Observation
#from rlbench.backend.task import Task
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation as R
import math

ACTION_SPACE_LIST = ["joint","jointgripper"]
STATE_SPACE_LIST = ["xyz"]
REWARD_TYPE_LIST = ['sparse']

class RLBenchEnvJoint:
    def __init__(self,config):
        self.config = config
        self.reward_shaping_type = config['environment']['reward']['reward_shaping_type']
        self.reward_scalor = config['environment']['reward']['reward_scalor']
        self.reward_bonus = config['environment']['reward']['reward_bonus']
        self.quat = np.array([0,1,0,0])
        self.quat = self.quat / np.linalg.norm(self.quat)
        self.obs_dim = config['environment']['obs_dim']
        self.action_space = config['agent']['action_space']
        self.task_name = self.config['environment']['task']['name']
        self.task_params = self.config['environment']['task']['params']
   
        self.act_dim = config['environment']['act_dim']
        self.boundary_min = np.array(config['agent']['boundary_min'])[:self.act_dim]
        self.boundary_max = np.array(config['agent']['boundary_max'])[:self.act_dim]
        self.state_space = config['environment']['state_space']

        assert self.action_space in ACTION_SPACE_LIST
        assert self.reward_shaping_type in REWARD_TYPE_LIST

        if self.config['environment']['camera'] == True:
            cam_config = CameraConfig(rgb=True, depth=True, point_cloud=True, mask=False,image_size=(256, 256),
                                    render_mode=RenderMode.OPENGL)
            obs_config = ObservationConfig()
            obs_config.set_all(False)
            obs_config.joint_positions = False
            obs_config.gripper_pose = True
            obs_config.right_shoulder_camera = cam_config
            obs_config.left_shoulder_camera = cam_config
            obs_config.wrist_camera = cam_config
            obs_config.front_camera = cam_config
            #obs_config.task_low_dim_state=True
        
        else:
            obs_config = ObservationConfig()
            obs_config.set_all(False)
            obs_config.joint_positions = False
            obs_config.gripper_pose = True
            #obs_config.task_low_dim_state=True

        arm_action_mode = JointVelocity()

        gripper_action_mode = Discrete()

        act_mode = MoveArmThenGripper(arm_action_mode,gripper_action_mode)

        self.env = Environment(action_mode = act_mode, obs_config= obs_config,headless = self.config['environment']['headless'], robot_setup = 'ur3baxter')

        self.env.launch()

        self.task_env = self.env.get_task(self.env._string_to_task(self.task_name +'.py'))

        self.reset()
        
    
    def shuttdown(self):
        self.reset()
        self.env.shutdown()
    
    def reset_once(self):
        o = self.task_env.reset()
        o = self.get_obs()
        return o

    def reset(self):

        while True:
            try:
                o = self.task_env.reset()
                if self.init_state_valid():
                    break
                else:
                    print("Init state is not valid. Repeat reset.")
            except:
                print("Could not reset the environment. Repeat reset.")
                time.sleep(1)

        o = self.get_obs()
        return o 
    
    def init_state_valid(self):
        return True
                
    
    def step(self,a_model):
  
        if self.action_space == "joint": a_model = np.append(a_model, [1.0])

        o, r, d, info = self.task_env.step(a_model)
        ## Get the observation
        o = self.get_obs()

        return o, r, d, info
    
    def get_max_return(self):
        return None

    def get_obs(self):
        if self.task_name == "reach_target_no_distractors":
            if self.action_space == "xyz":
                return self.task_env._scene.task._target_place.get_position()[:self.obs_dim]
       
