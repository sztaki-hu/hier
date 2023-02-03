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

        for _ in tqdm(range(self.demo_num), desc ="Loading demos: ", colour="green"):  
            o = self.env.reset()
            episode_transitions = []
            ret = 0
            for i in range(int(self.config['environment']['task']['params'][0])):
                block_index_x = i * 3
                block_index_y = i * 3 + 1
                block_index_z = i * 3 + 2
                target_index_x = 4 * 3
                target_index_y = 4 * 3 + 1
                target_index_z = 4 * 3 + 2
                #print(o)
                a = o[[block_index_x,block_index_y,block_index_z,target_index_x,target_index_y,target_index_z]]
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
        self.logger.save_demos(self.demo_name,  self.demo_buffer)

        self.env.shuttdown()

        # batch = self.demo_buffer.sample_batch(self.batch_size) 
        # print(batch)
        # print(self.demo_buffer.ptr)
    

    def load_demos(self):
        return self.logger.load_demos(self.demo_name)