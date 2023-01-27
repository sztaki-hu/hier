import os
import math
import numpy as np
import torch
import time

from tqdm import tqdm

from rltrain.buffers.replay import ReplayBuffer
from rltrain.agents.agent import Agent

from rltrain.envs.builder import make_env

class Tester:

    def __init__(self,agent,logger,config):
        #self.env = env
        self.agent = agent
        self.logger = logger
        self.config = config

        self.seed = config['general']['seed']        
        self.max_ep_len = config['sampler']['max_ep_len']

        self.num_test_episodes = config['tester']['num_test_episodes'] 
        self.max_ep_len = config['sampler']['max_ep_len']  

        self.act_dim = config['environment']['act_dim']
    
    def eval_range(self,epoch):
        if self.act_dim != 1: return
        inputs_np = np.linspace(self.boundary_min[0], self.boundary_max[0], num=10, endpoint=True)
        inputs = torch.from_numpy(inputs_np.astype(np.float32)).to(self.device)
        inputs = inputs.view(-1,1)

        outputs, _ = self.agent.ac.pi(inputs, deterministic = True, with_logprob = False)
        outputs_np = outputs.cpu().detach().numpy().flatten()

        data = np.vstack((inputs_np, outputs_np)).T
        self.logger.save_eval_range(data, epoch)
    
    def display_agent(self,model_name, num):
        path = self.logger.get_model_path(model_name)
        self.agent.load_weights(path)

        avg_return = self.start(verbose = True)

        print("########################################")
        print("avg return: " + str(avg_return))
    
    def get_avg_return(self):
        return self.avg_return
    
    def save_result(self,t,epoch,avg_test_return):
        log_text = "AVG test return: " + str(epoch) + ". epoch ("+ str(t+1) + " transitions) : " + str(avg_test_return)
        tqdm.write(log_text) 
        self.logger.tb_writer_add_scalar("test/average_return", avg_test_return, epoch)
        
        #self.logger.save_model(self.agent.ac.pi.state_dict(),epoch)

    def start(self, epoch = None, verbose = False):

        avg_return = -1

        self.env = make_env(self.config)

        sum_return = 0
        for j in tqdm(range(self.num_test_episodes), desc ="Testing: ", leave=False):
            o, d, ep_ret, ep_len = self.env.reset(), False, 0, 0
            while not(d or (ep_len == self.max_ep_len)):
                # Take deterministic actions at test time 
                try:
                    a = self.agent.get_action(o, True)
                    #print(a)
                    o, r, d, _ = self.env.step(a)
                    ep_ret += r
                    ep_len += 1
                except:
                    tqdm.write("Error in simulation (test time), thus reseting the environment")
                    break               
            sum_return += ep_ret
            if verbose:
                tqdm.write("------------------------")
                tqdm.write("Obs: " + str(o) + " | Act: " + str(a))
                tqdm.write("Ep Ret: " + str(ep_ret) + " | Ep Len: " + str(ep_len))
        avg_return = sum_return / float(self.num_test_episodes)
        if epoch is not None:
            self.eval_range(epoch)
        
        self.env.shuttdown() 

        return avg_return

        