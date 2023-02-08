import numpy as np
import time 

from rlbench.observation_config import ObservationConfig, CameraConfig
from pyrep.const import RenderMode

from rlbench.environment import Environment
from rlbench.action_modes.action_mode import ActionMode, MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import ArmActionMode, JointPosition, EndEffectorPoseViaPlanning, EndEffectorPoseViaIK
from rlbench.action_modes.gripper_action_modes import GripperActionMode, Discrete

from rlbench.task_environment import TaskEnvironment
from rlbench.backend.observation import Observation
#from rlbench.backend.task import Task
from pyquaternion import Quaternion

ACTION_SPACE_LIST = ["xyz","pick_and_place_2d","pick_and_place_3d"]

class RLBenchEnv:
    def __init__(self,config):
        self.config = config
        self.reward_shaping_use = config['environment']['reward']['reward_shaping_use']
        self.reward_shaping_type = config['environment']['reward']['reward_shaping_type']
        self.quat = np.array([0,1,0,0])
        self.quat = self.quat / np.linalg.norm(self.quat)
        self.obs_dim = config['environment']['obs_dim']
        self.action_space = config['agent']['action_space']
        self.task_name = self.config['environment']['task']['name']
        self.task_params = self.config['environment']['task']['params']
        self.target_blocks_num = self.task_params[0]
        self.act_dim = config['environment']['act_dim']
        self.boundary_min = np.array(config['agent']['boundary_min'])[:self.act_dim]
        self.boundary_max = np.array(config['agent']['boundary_max'])[:self.act_dim]

        #self.desk_z = 0.765
        #self.tower_z = 0.765

        assert self.action_space in ACTION_SPACE_LIST

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

        arm_action_mode = EndEffectorPoseViaPlanning()

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
                break
            except:
                print("Could not reset the environment. Repeat reset.")
                time.sleep(1)
        #o = self.task_env.reset()
        o = self.get_obs()
        return o 
    
    def step(self,a_model):
        if self.action_space == "xyz":
            a = self.model2robot_xyz(a_model)
            o, r, d, info = self.task_env.step(a)    
        # elif self.action_space == "pick_and_place_2d":
        #     poses = self.model2robot_pick_and_place_2d(a_model)
            o, r, d, info = self.execute_path(poses)  
        elif self.action_space == "pick_and_place_3d":
            poses = self.model2robot_pick_and_place_3d(a_model)
            o, r, d, info = self.execute_path(poses)    
        r = 100 * r
        o = self.get_obs()
        if self.out_of_bound_check(o):
            d = 1
            info = {'code': -11, 'description': 'Block is out of bounds ' + str(o)}
        if self.reward_shaping_use:
            if self.reward_shaping_type == 'mse':
                r = self.reward_shaping_mse(a)
        return o, r, d, info
    
    def reward_shaping_mse(self,o):
        return -((o - self.get_obs())**2).sum()

    def get_obs(self):
        if self.action_space == "xyz":
            return self.task_env._scene.task._target_place.get_position()[:self.obs_dim]
        elif self.action_space == "pick_and_place_3d":
            obs = [item.get_position() for item in self.task_env._scene.task._observation]
            obs = np.hstack(obs)
            return obs

            # obj = self.task_env._scene.task._graspable_objects[0].get_position()[:2]
            # trgt = self.task_env._scene.task._target_place.get_position()[:2]
            # return np.concatenate([obj,trgt])
    
    def model2robot_pick_and_place_2d(self,a):
        xy1 = a[:2]
        xy2 = a[2:]
        xyz1 = np.concatenate([xy1,np.array([self.desk_z])])
        xyz2 = np.concatenate([xy2,np.array([self.desk_z])])
        return self.pick_and_place_planner(xyz1, self.quat, xyz2, self.quat, h1 = None, d1 = None, h2 = None, d2 = None)
    
    def model2robot_pick_and_place_3d(self,a):
        xyz1 = a[:3]
        xyz2 = a[3:]
        z = max(xyz1[2],xyz2[2])
        d = 0.03 * 1.5
        h = z + d
        return self.pick_and_place_planner(xyz1, self.quat, xyz2, self.quat, h1 = h, d1 = None, h2 = h, d2 = None)


    def model2robot_xyz(self,a_model):
        if self.obs_dim == 3:
            return np.concatenate([a_model,self.quat,np.array([1])])
        else:
            obs_raw = self.task_env._scene.task._target_place.get_position()
            return np.concatenate([a_model,obs_raw[self.obs_dim:],self.quat,np.array([1])])

    def pyquat2rlbench(self,quat): # (w,x,y,z) --> (x,y,z,w)
        return np.array([quat[1], quat[2], quat[3], quat[0]])
    def rlbench2pyquat(self,quat): # (x,y,z,w) --> (w,x,y,z)
        return Quaternion(quat[3], quat[0], quat[1], quat[2])

    def grasp_release_planner(self, xyz, quat, grasp = True, drop = False, h = None, d = None):

        qn = quat / np.linalg.norm(quat)
        x = xyz[0]
        y = xyz[1]
        z = xyz[2]

        if d is None: d = 0.03 * 2
        if h is None: h = z + d

        poses = []
        if grasp == False and drop == True:
            poses.append(np.array([x,y,h,qn[0],qn[1],qn[2],qn[3],1]))
        else:
            poses.append(np.array([x,y,h,qn[0],qn[1],qn[2],qn[3],grasp]))
            poses.append(np.array([x,y,z,qn[0],qn[1],qn[2],qn[3],grasp]))
            poses.append(np.array([x,y,z,qn[0],qn[1],qn[2],qn[3],not grasp]))
            poses.append(np.array([x,y,h,qn[0],qn[1],qn[2],qn[3],not grasp]))
    
        return poses

    def pick_and_place_planner(self,xyz1, quat1, xyz2, quat2, h1 = None, d1 = None, h2 = None, d2 = None):

        #pose_ref = [np.array([0.25,0,self.tower_z,self.quat[0],self.quat[1],self.quat[2],self.quat[3],1])]
        poses1 = self.grasp_release_planner(xyz1, quat1, grasp = True, drop = False, h = h1 , d = d1)
        poses2 = self.grasp_release_planner(xyz2, quat2, grasp = False, drop = False, h = h2 , d = d2)

        #return pose_ref + poses1 + poses2
        return poses1 + poses2

    def pick_and_drop_planner(self,xyz1, quat1, xyz2, quat2, h1 = None, d1 = None, h2 = None, d2 = None):

        poses1 = self.grasp_release_planner(xyz1, quat1, grasp = True, drop = False, h = h1 , d = d1)
        poses2 = self.grasp_release_planner(xyz2, quat2, grasp = False, drop = True, h = h2 , d = d2)

        return poses1 + poses2

    def slide_block(self,rob_pose,xyz1,xyz2,quat):

        qn = quat / np.linalg.norm(quat)

        poses = []
        poses.append(np.array([rob_pose[0],rob_pose[1],rob_pose[2],rob_pose[3],rob_pose[4],rob_pose[5],rob_pose[6],0]))
        poses.append(np.array([xyz1[0],xyz1[1],xyz1[2],qn[0],qn[1],qn[2],qn[3],0]))
        poses.append(np.array([xyz2[0],xyz2[1],xyz1[2],qn[0],qn[1],qn[2],qn[3],0]))

        return poses

    def execute_path(self,poses):
        for pos in poses:
            observation, reward, done, info = self.task_env.step(pos)
        return observation, reward, done, info
    
    def out_of_bound_check(self,o):
        if self.task_name == "stack_blocks":
            for j in range(1,self.target_blocks_num+1):
                block_index =  (j * 3, j * 3 + 1, j * 3 + 2)
                block = o[[block_index[0],block_index[1],block_index[2]]]
                if np.any(np.greater(block,self.boundary_max[:3])):
                    return True
                if np.any(np.less(block,self.boundary_min[:3])):
                    return True
        return False
                 

