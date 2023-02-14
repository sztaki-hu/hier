import numpy as np
import torch
import time
from tqdm import tqdm

from rltrain.buffers.replay import ReplayBuffer
from rltrain.envs.builder import make_env

class Demo:
    def __init__(self,logger,config):
        self.logger = logger
        self.config = config  
            

        self.demo_use = config['demo']['demo_use']  
        self.demo_num = int(config['demo']['demo_buffer_size'])  
        self.demo_ratio = config['demo']['demo_ratio'] 
        self.demo_name = config['demo']['demo_name'] 
        self.demo_create = config['demo']['demo_create']
        self.demo_block_order = config['demo']['demo_block_order']

        self.obs_dim = config['environment']['obs_dim']
        self.act_dim = config['environment']['act_dim']
        self.action_space = config['agent']['action_space']
        self.gamma = config['agent']['gamma'] 
        self.n_step = config['agent']['n_step'] 
        self.batch_size = config['trainer']['batch_size'] 

        self.task_name = self.config['environment']['task']['name']
        self.target_blocks_num = int(self.config['environment']['task']['params'][0])
        self.distractor_blocks_num = int(self.config['environment']['task']['params'][1])
        self.tower_height = int(self.config['environment']['task']['params'][2])

        self.demo_buffer = ReplayBuffer(obs_dim=self.obs_dim, act_dim=self.act_dim, size=self.demo_num)

    def create_demos(self):  

        if self.action_space == "xyz":
            self.create_demos_xyz()
        # elif self.action_space == "pick_and_place_2d":
        #     self.create_demos_pick_and_place_2d()
        elif self.action_space == "pick_and_place_3d":
            if self.task_name == "stack_blocks":
                self.create_demos_stack_blocks_pick_and_place_3d()

    def create_demos_xyz(self):  

        self.env = make_env(self.config)

        for _ in tqdm(range(self.demo_num), desc ="Loading demos: ", colour="green"):  
            o = self.env.reset()
            a = o
            try:
                o2, r, d, info = self.env.step(a)
                if r > 0:
                     self.demo_buffer.store(o, a, r, o2, d)
                else:
                    tqdm.write("The demonstration is not successful, thus it is not added") 
            except:
                tqdm.write("Error in simulation, this demonstration is not added")         
        self.logger.save_demos(self.demo_name,  self.demo_buffer)

        self.env.shuttdown()

        # batch = self.demo_buffer.sample_batch(self.batch_size) 
        # print(batch)
        # print(self.demo_buffer.ptr)
        
    
    def create_demos_stack_blocks_pick_and_place_3d(self):  

        self.env = make_env(self.config)
        unsuccessful_num = 0

        assert self.target_blocks_num >= self.tower_height

        for _ in tqdm(range(self.demo_num), desc ="Loading demos: ", colour="green"):  
            while True:
                try:
                    o = self.env.reset_once()
                    if self.env.init_state_valid():
                        break
                    else:
                        tqdm.write('Init state is not valid. Repeat env reset.')
                        time.sleep(0.1)
                except:
                    tqdm.write('Could not reset the environment. Repeat env reset.')
                    time.sleep(1)

            ep_transitions = []
            ret = 0

            target_index =  (0, 1, 2)
            target = o[[target_index[0],target_index[1],target_index[2]]]

            blocks_init = []
            dists = []
            for j in range(1,self.target_blocks_num+1):
                block_index =  (j * 3, j * 3 + 1, j * 3 + 2)
                block = o[[block_index[0],block_index[1],block_index[2]]]
                dists.append(np.sum(np.square(target - block)))
                blocks_init.append(block)         
            
            if self.demo_block_order == "dist_based":
                blocks = []
                for _ in range(self.target_blocks_num):
                    index = np.argmin(dists)
                    blocks.append(blocks_init[index])
                    dists[index] = float('inf')
            elif self.demo_block_order == "random":
                print("Not implemented yet")
            elif self.demo_block_order == "as_init":
                blocks = blocks_init
            
            for i in range(self.tower_height):
                
                a = np.hstack((blocks[i],target)) 
                a[5] += 0.01 + 0.03 * i
                try:
                    o2, r, d, info = self.env.step(a)
                    ep_transitions.append((o, a, r, o2, d))
                    o = o2
                    ret += r
                except:
                    tqdm.write("Error in simulation, this demonstration is not added")
                    ret = -1
                    break
            if ret > 0:
                for i in range(len(ep_transitions)):
                    o, a, r, o2, d = ep_transitions[i]
                    r_nstep = ep_transitions[i][2]
                    obs_nstep = ep_transitions[i][3]
                    d_nstep = ep_transitions[i][4]
                    for j in range(1,self.n_step):
                        if d_nstep == 0 and i + j < len(ep_transitions):
                            r_nstep += ep_transitions[i+j][2] * self.gamma**j
                            obs_nstep = ep_transitions[i+j][3]
                            d_nstep = ep_transitions[i+j][4]
                        else:
                            break
                    n_nstep = j
                    self.demo_buffer.store(o, a, r, o2, d, r_nstep, obs_nstep, d_nstep, n_nstep)

            else:
                unsuccessful_num += 1   
                tqdm.write("The demonstration is not successful, thus it is not added " + str(unsuccessful_num))    
        
        self.env.shuttdown()

        self.logger.save_demos(self.demo_name,  self.demo_buffer)

        # batch = self.demo_buffer.sample_batch(self.batch_size) 
        # print(batch)
        # print(self.demo_buffer.ptr)
    
    def clean_up_old_demo(self):
        if self.demo_exists():
            self.logger.remove_old_demo(self.demo_name)
            print("Old demo removed")
            

    def demo_exists(self):
        return self.logger.demo_exists(self.demo_name)

    def load_demos(self):
        return self.logger.load_demos(self.demo_name)