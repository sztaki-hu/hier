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
        self.env_name = config['environment']['name']
        self.env_headless = self.config['environment']['headless']

        self.epochs = config['trainer']['epochs'] 
        self.steps_per_epoch = config['trainer']['steps_per_epoch'] 

        self.agent_num = int(config['agent']['agent_num'])
    
    def eval_range(self,epoch):
        if self.act_dim != 1: return
        inputs_np = np.linspace(self.boundary_min[0], self.boundary_max[0], num=10, endpoint=True)
        inputs = torch.from_numpy(inputs_np.astype(np.float32)).to(self.device)
        inputs = inputs.view(-1,1)

        outputs, _ = self.agent.ac.pi(inputs, deterministic = True, with_logprob = False)
        outputs_np = outputs.cpu().detach().numpy().flatten()

        data = np.vstack((inputs_np, outputs_np)).T
        self.logger.save_eval_range(data, epoch)
    
    def get_avg_return(self):
        return self.avg_return
    
    def save_result(self,t,epoch,avg_test_return):
        log_text = "AVG test return: " + str(epoch) + ". epoch ("+ str(t+1) + " transitions) : " + str(avg_test_return)
        tqdm.write(log_text) 
        self.logger.tb_writer_add_scalar("test/average_return", avg_test_return, t)
        
        #self.logger.save_model(self.agent.ac.pi.state_dict(),epoch)
    
    def np2dict(self,data_np):
        data = {}
        for agent_id in range(data_np.shape[0]):
            data["agent_" + str(agent_id)]= data_np[agent_id]        
        return data
    
    def start(self,end_flag,test2train):
        
        self.env = make_env(self.config)
        self.test2train = test2train

        max_return = self.env.get_max_return()

        avg_return_np = np.zeros(self.agent_num)
        succes_rate_np = np.zeros(self.agent_num)
        avg_episode_len_np = np.zeros(self.agent_num)
    
        epochs = np.zeros(self.agent_num)
        epoch = 1
        while epoch <= self.epochs:
            trainer_ended = end_flag.value

            for agent_id in range(self.agent_num):
            
                if epoch == epochs[agent_id] + 1:
                    model_name = self.logger.get_model_epoch(epoch,agent_id)
                
                if model_name != None:

                    path = self.logger.get_model_path(model_name,agent_id)
                    
                    while True:
                        try:
                            self.agent.load_weights(path,mode="pi",eval=True)
                            break
                        except:
                            data = {'code': -31, 'agent_id': agent_id, 'description':'Tester could not open weight file'}
                            test2train.put(data)
                            time.sleep(1.0)

                    avg_return, succes_rate, avg_episode_len, error_in_env, out_of_bounds = self.test_v2(max_return)
                    avg_return_np[agent_id] = avg_return
                    succes_rate_np[agent_id] = succes_rate
                    avg_episode_len_np[agent_id] = avg_episode_len
                    data = {'code': 1, 'agent_id': agent_id, 'avg_return': avg_return, 'succes_rate': succes_rate,'error_in_env': error_in_env, 'avg_episode_len': avg_episode_len, 'out_of_bounds':out_of_bounds, 'epoch': epoch, 'description':'Average test result'}
                    test2train.put(data)

                    epochs[agent_id] += 1

            if np.all(epochs == epoch):
                t = epoch * self.steps_per_epoch 
                for i in range(self.agent_num):
                    self.logger.tb_writer_add_scalar("test/test_ret_"+str(i), avg_return_np[i], t) 
                    self.logger.tb_writer_add_scalar("test/test_ep_len_"+str(i), avg_episode_len_np[i], t)
                    if succes_rate is not None: self.logger.tb_writer_add_scalar("test/test_success_"+str(i), succes_rate_np[i], t)
                epoch += 1

            if trainer_ended == True:
                break
            
            time.sleep(1.0)
        
        self.env.shuttdown() 

    def reset_env(self):
        while True:
            try:
                o = self.env.reset_once()
                if self.env.init_state_valid():
                    break
                else:
                    #data = {'code': -21, 'description':'Init state is not valid. Repeat env reset.'}
                    #self.test2train.put(data)
                    time.sleep(0.1)
            except:
                data = {'code': -1, 'description':'Could not reset the environment. Repeat env reset.'}
                self.test2train.put(data)
                time.sleep(1)
        
        return o
    
    def test_v2(self,max_return = None):
        
        avg_return = -1
        sum_return = 0
        sum_episode_len = 0
        
        error_in_env = 0
        out_of_bounds = 0
        
        for j in range(self.num_test_episodes):

            o, d, ep_ret, ep_len = self.reset_env(), False, 0, 0

            while not(d or (ep_len == self.max_ep_len)):
                # Take deterministic actions at test time 
                try:
                    a = self.agent.get_action(o, True)
                    o, r, d, info = self.env.step(a)
                except:
                    #data = {'code': -3, 'description':'[Test]: Error  simulation, thus reseting the environment'}
                    #self.test2train.put(data)
                    error_in_env+=1
                    break   

                if bool(info): 
                    if 'code' in info:            
                        if info['code'] < 0:
                            if info['code'] == -11:
                                #data = {'code': -11, 'description': '[Test]: Block is out of bounds'}
                                #self.test2train.put(data) 
                                out_of_bounds+=1 
                            break

                ep_ret += r
                ep_len += 1    
            sum_return += ep_ret
            sum_episode_len += ep_len        
        avg_return = sum_return / float(self.num_test_episodes)
        avg_episode_len = sum_episode_len / float(self.num_test_episodes)
        error_in_env = error_in_env / float(self.num_test_episodes)
        out_of_bounds = out_of_bounds / float(self.num_test_episodes)
        succes_rate = avg_return / float(max_return) if max_return is not None else None
        
        return avg_return, succes_rate, avg_episode_len, error_in_env, out_of_bounds

    def test(self, epoch = None, verbose = False):

        avg_return = -1

        self.env = make_env(self.config)

        sum_return = 0
        #for j in tqdm(range(self.num_test_episodes), desc ="Testing: ", leave=False):
        for j in range(self.num_test_episodes):
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
    
    def display_agent(self,model_name,num_display_episode):
        path = self.logger.get_model_path(model_name)
        self.agent.load_weights(path,mode="pi",eval=True)

        avg_return = self.display_test(num_display_episode)

        print("########################################")
        print("avg return: " + str(avg_return))
        return avg_return

    def test_agent_all(self,num_display_episode):

        self.env = make_env(self.config)
        max_return = self.env.get_max_return()

        avg_return_list = []
        succes_rate_list = []
        avg_episode_len_list = []
        error_in_env_list = []
        out_of_bounds_list = [] 

        #self.epochs = 1

        for model_i in tqdm(range(self.epochs), desc ="Testing: "):
            model_name = "model_"+str(model_i+1)
            path = self.logger.get_model_path(model_name)
            self.agent.load_weights(path,mode="pi",eval=True)

            avg_return = -1
            sum_return = 0
            sum_episode_len = 0
            
            error_in_env = 0
            out_of_bounds = 0

            for j in range(num_display_episode):

                o, d, ep_ret, ep_len = self.reset_env(), False, 0, 0

                while not(d or (ep_len == self.max_ep_len)):
                    # Take deterministic actions at test time 
                    try:
                        a = self.agent.get_action(o, True)
                        o, r, d, info = self.env.step(a)
                    except:
                        #data = {'code': -3, 'description':'[Test]: Error  simulation, thus reseting the environment'}
                        #self.test2train.put(data)
                        error_in_env+=1
                        break   

                    if bool(info): 
                        if 'code' in info:            
                            if info['code'] < 0:
                                if info['code'] == -11:
                                    #data = {'code': -11, 'description': '[Test]: Block is out of bounds'}
                                    #self.test2train.put(data) 
                                    out_of_bounds+=1 
                                break

                    ep_ret += r
                    ep_len += 1    
                sum_return += ep_ret
                sum_episode_len += ep_len        
            avg_return_list.append(sum_return / float(num_display_episode))
            avg_episode_len_list.append(sum_episode_len / float(num_display_episode))
            error_in_env_list.append(error_in_env / float(num_display_episode))
            out_of_bounds_list.append(out_of_bounds / float(num_display_episode))
            succes_rate_list.append(avg_return_list[-1] / float(max_return)) if max_return is not None else None

        self.env.shuttdown() 

        return avg_return_list, succes_rate_list, avg_episode_len_list, error_in_env_list, out_of_bounds_list

    def display_test(self, num_display_episode):

        avg_return = -1

        self.env = make_env(self.config)

        sum_return = 0
        for j in tqdm(range(num_display_episode), desc ="Testing: ", leave=False):
            
            [o, info], d, ep_ret, ep_len = self.env.reset_with_init_check(), False, 0, 0

            print(info)
            

            while not(d or (ep_len == self.max_ep_len)):

                # Take deterministic actions at test time 
                try:
                    
                    a = self.agent.get_action(o, True)
                    #print(a)
                    o, r, terminated, truncated, info = self.env.step(a)
                    d = terminated or truncated
                    # print(o)
                    # print(info)
                    time.sleep(0.01)

                except:
                    tqdm.write('[Test]: Error  simulation, thus reseting the environment')
                    break    

                if bool(info): 
                    if 'code' in info:            
                        if info['code'] < 0:
                            if info['code'] == -11:
                                tqdm.write('[Test]: Block is out of bounds')  
                            break

                ep_ret += r
                ep_len += 1   
                #tqdm.write(str(ep_len))       
            sum_return += ep_ret          
            tqdm.write("------------------------")
            #tqdm.write("Obs: " + str(o) + " | Act: " + str(a))
            tqdm.write("Ep Ret: " + str(ep_ret) + " | Ep Len: " + str(ep_len) + " | Info: " + str(info))
   
        avg_return = sum_return / float(num_display_episode)
        
        self.env.shuttdown() 

        return avg_return

        