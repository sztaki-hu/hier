import numpy as np
import time

#REWARD_TYPE_LIST = ['sparse','mse','envchange','subgoal']
REWARD_TYPE_LIST = ['sparse','subgoal']

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

        self.subgoal_level = 0
        #self.desk_height = 0.765
        self.block_on_desk_z = 0.765
        self.block_size = 0.03

        assert self.reward_shaping_type in REWARD_TYPE_LIST

        self.reset()

    def shuttdown(self):
        return None
    
    def reset_once(self):
        return self.reset()
    
    def reset(self):
        if self.task_name == 'stack_blocks':
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
        else:
            self.observation = np.random.uniform(low=self.boundary_min, high=self.boundary_max, size=(self.obs_dim))
            return self.observation
    
    def step(self,a):
        if self.action_space == "pick_and_place_3d":
            # Movement of the block
            r = 0
            for i in range(self.task_params[0] + self.task_params[1]):
                block = self.observation[(i+1)*3:(i+2)*3]
                if np.allclose(a[:3], block, rtol=0.0, atol=0.01, equal_nan=False):
                    self.observation[(i+1)*3:(i+2)*3] = a[3:6]
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
        
        else:
            d = np.allclose(a[:self.obs_dim], self.observation, rtol=0.0, atol=0.02, equal_nan=False)
            r = float(d) * 100
            info = None
            if self.reward_shaping_use:
                if self.reward_shaping_type == 'mse':
                    r = self.reward_shaping_mse(o)
            # avg = (self.boundary_min[0] + self.boundary_max[0]) / 2.0
            # range = abs((self.boundary_max[0] - self.boundary_min[0]))
            # o = (o - avg) / range
            o = self.observation
            time.sleep(0.05)
            return o, r, d, info

    def reward_shaping_subgoal_stack_blocks(self,o):
        
        target_index =  (0, 1, 2)
        target = o[[target_index[0],target_index[1],target_index[2]]]

        blocks = []
        #dists = []
        for j in range(1,self.task_params[0]+1):
            block_index =  (j * 3, j * 3 + 1, j * 3 + 2)
            block = o[[block_index[0],block_index[1],block_index[2]]]
            #dists.append(np.sum(np.square(target - block)))
            blocks.append(block) 
        
        for i in range(self.subgoal_level+1):
            subsubgoal_reached = False
            target[2] = self.block_on_desk_z + i * self.block_size
            for block in blocks:
                if np.allclose(target, block, rtol=0.00, atol=0.01, equal_nan=False):
                    subsubgoal_reached = True
                    break
            if subsubgoal_reached == False: 
                return 0
        
        self.subgoal_level += 1
        return self.reward_bonus * self.subgoal_level

    def init_state_valid(self):
        if self.task_name == "stack_blocks":
            o = self.observation
            target_index =  (0, 1, 2)
            target = o[[target_index[0],target_index[1],target_index[2]]]
            for j in range(1,self.target_blocks_num+1):
                block_index =  (j * 3, j * 3 + 1, j * 3 + 2)
                block = o[[block_index[0],block_index[1],block_index[2]]]
                if np.allclose(block, target, rtol=0.0, atol=0.02, equal_nan=False):
                    return False
        return True
        
    def reward_shaping_mse(self,o):
        return -((o - self.observation)**2).sum()
    
    def get_target(self): #for demos
        return self.observation


