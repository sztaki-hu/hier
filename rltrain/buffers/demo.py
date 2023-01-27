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

        self.demo_buffer = ReplayBuffer(obs_dim=self.obs_dim, act_dim=self.act_dim, size=self.demo_num)

    def create_demos(self):  

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
    
    def load_demos(self):
        return self.logger.load_demos(self.demo_name)