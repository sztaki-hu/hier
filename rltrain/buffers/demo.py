import numpy as np
import torch
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
        self.obs_dim = config['environment']['obs_dim']
        self.act_dim = config['environment']['act_dim']
        self.action_space = config['agent']['action_space']
        self.batch_size = config['trainer']['batch_size'] 

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
            self.create_demos_pick_and_place_3d()

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
    
    def create_demos_pick_and_place_3d(self):  

        self.env = make_env(self.config)
        unsuccessful_num = 0

        assert self.target_blocks_num >= self.tower_height

        for _ in tqdm(range(self.demo_num), desc ="Loading demos: ", colour="green"):  
            o = self.env.reset()
            episode_transitions = []
            ret = 0

            print(".............................")
            print(o.shape)
            print(o)

            target_index =  (0, 1, 2)
            target = o[[target_index]][0]

            blocks = []
            dists = []
            for j in range(1,self.target_blocks_num+1):
                block_index =  (j * 3, j * 3 + 1, j * 3 + 2)
                block = o[[block_index]][0]
                dists.append(np.sum(np.square(target - block)))
                blocks.append(block)                
            
            blocks_arranged = []
            for _ in range(self.target_blocks_num):
                index = np.argmin(dists)
                blocks_arranged.append(blocks[index])
                dists[index] = float('inf')

            for i in range(self.tower_height):
                
                a = np.hstack((blocks_arranged[i],target)) 
                a[5] += 0.01 + 0.03 * i
                try:
                    o2, r, d, info = self.env.step(a)
                    episode_transitions.append((o, a, r, o2, d))
                    o = o2
                    ret += r
                except:
                    tqdm.write("Error in simulation, this demonstration is not added")
                    ret = -1
                    break
            if ret > 0:
                for t in episode_transitions:
                    self.demo_buffer.store(t[0],t[1],t[2],t[3],t[4])
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