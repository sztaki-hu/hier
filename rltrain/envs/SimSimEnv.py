import numpy as np
import time
from pyquaternion import Quaternion

#REWARD_TYPE_LIST = ['sparse','mse','envchange','subgoal']
REWARD_TYPE_LIST = ['sparse','subgoal']
TASK_TYPE_LIST = ['stack_blocks']
ACTION_SPACE_LIST = ['pick_and_place_2d','pick_and_place_3d','pick_and_place_3d_quat','pick_and_place_3d_z90']
STATE_SPACE_LIST = ["xyz","xyz_quat","xyz_z90"]

class SimSimEnv:
    def __init__(self,config):
        self.config = config
        self.reward_shaping_type = config['environment']['reward']['reward_shaping_type']
        self.reward_scalor = config['environment']['reward']['reward_scalor']
        self.reward_bonus = config['environment']['reward']['reward_bonus']
        self.obs_dim = config['environment']['obs_dim']
        self.act_dim = config['environment']['act_dim']
        self.boundary_min = np.array(config['agent']['boundary_min'])[:self.act_dim]
        self.boundary_max = np.array(config['agent']['boundary_max'])[:self.act_dim]
        self.observation = np.zeros(self.obs_dim)

        self.task_name = config['environment']['task']['name']
        self.task_params = self.config['environment']['task']['params']
        self.target_blocks_num = self.task_params[0]
        self.action_space = config['agent']['action_space']
        self.state_space = config['environment']['state_space']

        self.subgoal_level = 0
        #self.desk_height = 0.765
        self.block_on_desk_z = 0.765
        self.block_size = 0.03

        assert self.reward_shaping_type in REWARD_TYPE_LIST
        assert self.task_name in TASK_TYPE_LIST
        assert self.action_space in ACTION_SPACE_LIST
        assert self.state_space in STATE_SPACE_LIST

        if self.state_space == "xyz":   
            self.obs_period = 3
        elif self.state_space == "xyz_quat":  
            self.obs_period = 7
        elif self.state_space == "xyz_z90":   
            self.obs_period = 4
        

        self.reset()

    def shuttdown(self):
        return None
    
    def reset_once(self):
        return self.reset()
    
    def reset(self):
        if self.task_name == 'stack_blocks':
            if self.state_space == "xyz":             
                obs = []
                obs.append(0.25)
                obs.append(0.0)
                obs.append(0.752)
                for _ in range(self.task_params[0] + self.task_params[1]):
                    block = np.random.uniform(low=self.boundary_min, high=self.boundary_max, size=(self.act_dim))
                    obs.append(block[0])
                    obs.append(block[1])
                    obs.append(0.765)
                self.observation = np.asarray(obs)
                self.subgoal_level = 0
                #print(self.observation)
                return self.observation.copy()
            elif self.state_space == "xyz_quat":              
                obs = []
                obs.append(0.25)
                obs.append(0.0)
                obs.append(0.752)
                obs.append(0.0)
                obs.append(0.0)
                obs.append(0.0)
                obs.append(1.0)
                for _ in range(self.task_params[0] + self.task_params[1]):
                    block = np.random.uniform(low=self.boundary_min[:2], high=self.boundary_max[:2], size=(2))
                    obs.append(block[0])
                    obs.append(block[1])
                    obs.append(0.765)

                    q = self.rlbench2pyquat(np.array([0,0,0,1])) # (x,y,z,w) --> (w,x,y,z)
                    z_rand = Quaternion(axis=[0, 0, 1], angle=np.random.uniform(low=0.0, high=3.14159265 / 4.0, size=1) ) # Rotate around Z
                    q2 = q * z_rand
                    quat_rand = self.pyquat2rlbench(q2) # (w,x,y,z) --> (x,y,z,w)  

                    obs.append(quat_rand[0])
                    obs.append(quat_rand[1])
                    obs.append(quat_rand[2])
                    obs.append(quat_rand[3])

                self.observation = np.asarray(obs)
                self.subgoal_level = 0
                #print(self.observation)
                return self.observation.copy()
            elif self.state_space == "xyz_z90":              
                obs = []
                obs.append(0.25)
                obs.append(0.0)
                obs.append(0.752)
                obs.append(0.0)

                for _ in range(self.task_params[0] + self.task_params[1]):
                    block = np.random.uniform(low=self.boundary_min[:2], high=self.boundary_max[:2], size=(2))
                    obs.append(block[0])
                    obs.append(block[1])
                    obs.append(0.765)

                    z = np.random.uniform(low=self.boundary_min[3], high=self.boundary_max[3], size=(1))[0]
                    obs.append(z)

                self.observation = np.asarray(obs)
                self.subgoal_level = 0
                #print(self.observation)
                return self.observation.copy()
        else:
            self.observation = np.random.uniform(low=self.boundary_min, high=self.boundary_max, size=(self.obs_dim))
            return self.observation
    
    def step(self,a):
        if self.action_space == "pick_and_place_2d":
            # Movement of the block
            r = 0
            for i in range(self.task_params[0] + self.task_params[1]):
                block = self.observation[(i+1)*3:(i+2)*3]
                if np.allclose(a[:2], block[:2], rtol=0.0, atol=0.01, equal_nan=False):
                    self.observation[(i+1)*3:(i+2)*3] = np.array([a[2],a[3],self.block_on_desk_z + self.subgoal_level * self.block_size])
                    break
            o = self.observation.copy()
        elif self.action_space == "pick_and_place_3d":
            # Movement of the block
            r = 0
            for i in range(self.task_params[0] + self.task_params[1]):
                block = self.observation[(i+1)*3:(i+2)*3]
                if np.allclose(a[:3], block, rtol=0.0, atol=0.01, equal_nan=False):
                    self.observation[(i+1)*3:(i+2)*3] = a[3:6]
                    break
            o = self.observation.copy()
        elif self.action_space == "pick_and_place_3d_quat":
            # Movement of the block
            r = 0
            for i in range(self.task_params[0] + self.task_params[1]):
                block = self.observation[(i+1)*7:(i+2)*7]
                
                q = self.rlbench2pyquat(block[3:7]) # (x,y,z,w) --> (w,x,y,z)
                y_180 = Quaternion(axis=[0, 1, 0], angle=3.14159265) # Rotate 180 about Y
                q2 = q * y_180
                block[3:7] = self.pyquat2rlbench(q2) # (w,x,y,z) --> (x,y,z,w)
                block2 = np.copy(block)
                block2[3:7] = - block2[3:7]


                if np.allclose(a[:7], block, rtol=0.0, atol=0.01, equal_nan=False) or np.allclose(a[:7], block2, rtol=0.0, atol=0.01, equal_nan=False):
                    self.observation[(i+1)*7:(i+2)*7] = a[7:14]

                    q = self.rlbench2pyquat(a[10:14]) # (x,y,z,w) --> (w,x,y,z)
                    y_180 = Quaternion(axis=[0, 1, 0], angle=3.14159265) # Rotate 180 about Y
                    q2 = q * y_180
                    a[10:14] = self.pyquat2rlbench(q2) # (w,x,y,z) --> (x,y,z,w)  

                    break
            o = self.observation.copy()

        elif self.action_space == "pick_and_place_3d_z90":
            # Movement of the block
            r = 0
            for i in range(self.task_params[0] + self.task_params[1]):
                block = self.observation[(i+1)*self.obs_period:(i+2)*self.obs_period]
                if np.allclose(a[:self.obs_period], block, rtol=0.0, atol=0.01, equal_nan=False):
                    self.observation[(i+1)*self.obs_period:(i+2)*self.obs_period] = a[self.obs_period:2*self.obs_period]
                    break
            o = self.observation.copy()

        # Get reward
        bonus = self.reward_shaping_subgoal_stack_blocks(o)  
        if self.subgoal_level == self.task_params[2]:
            r = 1
            d = 1
        else:
            r = 0
            d = 0            
        if self.reward_shaping_type == 'sparse':
            r = r * self.reward_scalor
        elif self.reward_shaping_type == 'subgoal':        
            r = (r + bonus) * self.reward_scalor
        
        return o, r, d, None                  
        
        # else:
        #     d = np.allclose(a[:self.obs_dim], self.observation, rtol=0.0, atol=0.02, equal_nan=False)
        #     r = float(d) * 100
        #     info = None
        #     if self.reward_shaping_use:
        #         if self.reward_shaping_type == 'mse':
        #             r = self.reward_shaping_mse(o)
        #     # avg = (self.boundary_min[0] + self.boundary_max[0]) / 2.0
        #     # range = abs((self.boundary_max[0] - self.boundary_min[0]))
        #     # o = (o - avg) / range
        #     o = self.observation
        #     time.sleep(0.05)
        #     return o, r, d, info


    def reward_shaping_subgoal_stack_blocks(self,o):

        if self.obs_period == 3:
            target_index =  (0, 1, 2)
            target = o[[target_index[0],target_index[1],target_index[2]]]

            blocks = []
            #dists = []
            for j in range(1,self.task_params[0]+1):
                block_index =  (j * self.obs_period, j * self.obs_period + 1, j * self.obs_period + 2)
                block = o[[block_index[0],block_index[1],block_index[2]]]
                #dists.append(np.sum(np.square(target - block)))
                blocks.append(block) 
            
            subsubgoal_pts = 0
            for i in range(self.subgoal_level+1):        
                target[2] = self.block_on_desk_z + i * self.block_size
                for block in blocks:
                    if np.allclose(target, block, rtol=0.00, atol=0.01, equal_nan=False):
                        subsubgoal_pts += 1
                        break
                if i >= subsubgoal_pts: break
            if subsubgoal_pts == self.subgoal_level + 1: 
                subsubgoal_reached = True
                self.subgoal_level += 1
                return self.reward_bonus * self.subgoal_level
            
            return 0

        if self.obs_period == 4:
            target_index =  (0, 1, 2, 3)
            target = o[[target_index[0],target_index[1],target_index[2],target_index[3]]]

            blocks = []
            #dists = []
            for j in range(1,self.task_params[0]+1):
                block_index =  (j * self.obs_period, j * self.obs_period + 1, j * self.obs_period + 2, j * self.obs_period + 3)
                block = o[[block_index[0],block_index[1],block_index[2],block_index[3]]]
                #dists.append(np.sum(np.square(target - block)))
                blocks.append(block) 
            
            subsubgoal_pts = 0
            for i in range(self.subgoal_level+1):        
                target[2] = self.block_on_desk_z + i * self.block_size
                for block in blocks:
                    if np.allclose(target, block, rtol=0.00, atol=0.01, equal_nan=False):
                        subsubgoal_pts += 1
                        break
                if i >= subsubgoal_pts: break
            if subsubgoal_pts == self.subgoal_level + 1: 
                subsubgoal_reached = True
                self.subgoal_level += 1
                return self.reward_bonus * self.subgoal_level

            return 0

        if self.obs_period == 7:
            target_index =  (0, 1, 2, 3, 4, 5, 6)
            target = o[[target_index[0],target_index[1],target_index[2],target_index[3],target_index[4],target_index[5],target_index[6]]]


            blocks = []
            #dists = []
            for j in range(1,self.task_params[0]+1):
                block_index =  (j * self.obs_period, j * self.obs_period + 1, j * self.obs_period + 2,j * self.obs_period + 3, j * self.obs_period + 4, j * self.obs_period + 5,  j * self.obs_period + 6)
                block = o[[block_index[0],block_index[1],block_index[2],block_index[3],block_index[4],block_index[5],block_index[6]]]
                #dists.append(np.sum(np.square(target - block)))
                blocks.append(block) 
            
            subsubgoal_pts = 0
            for i in range(self.subgoal_level+1):
                target[2] = self.block_on_desk_z + i * self.block_size
                for block in blocks:
                    if np.allclose(target, block, rtol=0.00, atol=0.01, equal_nan=False):
                        subsubgoal_pts += 1
                        break
                if i >= subsubgoal_pts: break
                

            if subsubgoal_pts == self.subgoal_level + 1: 
                subsubgoal_reached = True
                self.subgoal_level += 1
                return self.reward_bonus * self.subgoal_level

            return 0


        

    def init_state_valid(self):
        if self.task_name == "stack_blocks":
            o = self.observation
            target_index =  (0, 1, 2)
            target = o[[target_index[0],target_index[1],target_index[2]]]
            for j in range(1,self.target_blocks_num+1):
                block_index =  (j * self.obs_period, j * self.obs_period + 1, j * self.obs_period + 2)
                block = o[[block_index[0],block_index[1],block_index[2]]]
                if np.allclose(block, target, rtol=0.0, atol=0.02, equal_nan=False):
                    return False
        return True
        
    def reward_shaping_mse(self,o):
        return -((o - self.observation)**2).sum()
    
    def get_target(self): #for demos
        return self.observation

    def get_max_return(self):
        if self.task_name == 'stack_blocks':
            if self.reward_shaping_type == 'sparse':
                return self.reward_scalor
            elif self.reward_shaping_type == 'subgoal': 
                bonus = 0
                for i in range(1,self.task_params[2]+1):
                    bonus += self.reward_bonus * i       
                return (1 + bonus) * self.reward_scalor
        return None

    def pyquat2rlbench(self,quat): # (w,x,y,z) --> (x,y,z,w)
        return np.array([quat[1], quat[2], quat[3], quat[0]])
    def rlbench2pyquat(self,quat): # (x,y,z,w) --> (w,x,y,z)
        return Quaternion(quat[3], quat[0], quat[1], quat[2])



