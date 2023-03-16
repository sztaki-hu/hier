import os
import math
import numpy as np
import torch
import time
import collections

from tqdm import tqdm

from multiprocessing import Process

from rltrain.buffers.replay import ReplayBuffer
from rltrain.agents.agent import Agent
from rltrain.runners.tester import Tester

DEMO_RATIO_TYPES = ['constant','linear_decay']

class Trainer:

    def __init__(self,device,demo_buffer,logger,config):

        self.device = device

        self.mode_sync = config['general']['sync'] 
        
        self.demo_buffer = demo_buffer
        self.logger = logger
        self.config = config

        self.seed = config['general']['seed'] 

        self.steps_per_epoch = config['trainer']['steps_per_epoch'] 
        self.epochs = config['trainer']['epochs'] 

        self.replay_size = int(config['buffer']['replay_buffer_size']) 
        self.batch_size = config['trainer']['batch_size'] 
        
        self.update_after = config['trainer']['update_after'] 
        self.update_every = config['trainer']['update_every'] 
        self.update_factor_np = np.array(config['trainer']['update_factor'])
        self.update_factor_max = np.max(self.update_factor_np)

        self.fallback_safety = config['trainer']['fallback_safety']['fb_bool'] 
        self.fb_th_type = config['trainer']['fallback_safety']['fb_th_type']
        self.fb_th_value = config['trainer']['fallback_safety']['fb_th_value'] 

        self.num_log_loss_points = config['logger']['num_log_loss_points'] 

        self.obs_dim = config['environment']['obs_dim']
        self.act_dim = config['environment']['act_dim']
        self.boundary_min = np.array(config['agent']['boundary_min'])[:self.act_dim]
        self.boundary_max = np.array(config['agent']['boundary_max'])[:self.act_dim]

        self.demo_use = config['demo']['demo_use']  
        self.demo_ratio_type = config['demo']['demo_ratio']['type']
        self.demo_ratio_params = config['demo']['demo_ratio']['params']

        assert self.demo_ratio_type in DEMO_RATIO_TYPES
        
        agent_num = int(config['agent']['agent_num'])         
        self.return_buffer_list = []
        self.episode_len_buffer_list = []
        self.ep_success_buffer_list = []
        for _ in range(agent_num):
            self.return_buffer_list.append(collections.deque(maxlen=20))
            self.episode_len_buffer_list.append(collections.deque(maxlen=20))
            self.ep_success_buffer_list.append(collections.deque(maxlen=20))


        self.heatmap_bool = config['logger']['heatmap']['bool']
        self.heatmap_res = config['logger']['heatmap']['resolution']

        
        self.heatmap_pick = np.zeros((self.heatmap_res, self.heatmap_res))
        self.heatmap_place = np.zeros((self.heatmap_res, self.heatmap_res))

        self.pretrain_bool = config['trainer']['pretrain']['bool']
        self.pretrain_factor = config['trainer']['pretrain']['factor']

        """
        Trainer

        Args:
            env : RLBench task-environment

            seed (int): Seed for random number generators.

            steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
                for the agent and the environment in each epoch.

            epochs (int): Number of epochs to run and train agent.

            replay_size (int): Maximum length of replay buffer.

            batch_size (int): Minibatch size for SGD.

            start_steps (int): Number of steps for uniform-random action selection,
                before running real policy. Helps exploration.

            update_after (int): Number of env interactions to collect before
                starting to do gradient descent updates. Ensures replay buffer
                is full enough for useful updates.

            update_every (int): Number of env interactions that should elapse
                between gradient descent updates. Note: Regardless of how long 
                you wait between updates, the ratio of env steps to gradient steps 
                is locked to 1.
            
            update_factor: The ration of gradient steps to env steps (overriding 
                the previous lock to 1).  

            num_test_episodes (int): Number of episodes to test the deterministic
                policy at the end of each epoch.

            max_ep_len (int): Maximum length of trajectory / episode / rollout.

            logger_kwargs (dict): Keyword args for EpochLogger.

            save_freq (int): How often (in terms of gap between epochs) to save
                the current policy and value function.

        """ 

    def get_fallback_th(self,checkpoint_test_return):
        if self.fb_th_type == "absolute":
            return checkpoint_test_return - self.fb_th_value
        elif self.fb_th_type == "relative":
            return checkpoint_test_return * self.fb_th_value
    
    def handle_messages(self,test2train,sample2train):

        msg = None
        avg_test_return = None
        agent_id = None

        if test2train.empty() == False:
            data = test2train.get()
            if data['code'] < 0:
                message = "Code: " + str(data['code']) + " Description: " + str(data['description'])
                tqdm.write("[warning]: " + message)  
                self.logger.print_logfile(message,level = "warning", terminal = False)  
                
            else:
                if data['code'] == 1:
                    avg_test_return = data['avg_return']
                    succes_rate = data['succes_rate']                      
                    epoch_test = data['epoch']
                    test_env_error = data['error_in_env']
                    test_out_of_bound = data['out_of_bounds']
                    avg_episode_len = data['avg_episode_len']
                    agent_id = data['agent_id']
                        
                    test_t = epoch_test * self.steps_per_epoch           
                    message = "[Test] AVG test return: " + str(epoch_test) + ". epoch ("+ str(test_t) + " transitions) | Agent id:" + str(agent_id) + " | Avg return: " + str(avg_test_return) + " | succes_rate: " + str(succes_rate) + " | avg episode len: " + str(avg_episode_len) + " | env error rate: " + str(test_env_error) + " | out of bound rate: " +str(test_out_of_bound)
                    tqdm.write("[info]: " + message)  
                    self.logger.print_logfile(message,level = "info", terminal = False)  
            msg = ('test',data['code'],avg_test_return,agent_id)
                
            
        elif sample2train.empty() == False:
            data = sample2train.get()
            if data['code'] < 0:                 
                message = "Code: " + str(data['code']) + " Description: " + str(data['description'])
                #tqdm.write("[warning]: " + message)  
                self.logger.print_logfile(message,level = "warning", terminal = False)  
                if int(data['code']) == -2:
                    self.env_error_num += 1 
                if int(data['code']) == -11:
                    self.out_of_bounds_num += 1 
            

            elif data['code'] == 11:
                #print(data)
                agent_id = data['agent_id']
                self.return_buffer_list[agent_id].append(float(data['ep_ret']))
                self.episode_len_buffer_list[agent_id].append(int(data['ep_len']))
                self.ep_success_buffer_list[agent_id].append(float(data['ep_success']))
                if self.heatmap_bool:  
                    if data['heatmap_pick'] is not None: self.heatmap_pick = data['heatmap_pick']
                    if data['heatmap_place'] is not None: self.heatmap_place = data['heatmap_place']

            elif data['code'] == 12:  
                self.reward_bonus_num+=1
            
            msg = ('sample',data['code'],None)
        
        return msg

    
    def get_batch(self,t,replay_buffer):
        
        if self.demo_use == False:
            return replay_buffer.sample_batch(self.batch_size), 0
        else:
            # get demo ratio
            if self.demo_ratio_type == 'constant':
                demo_ratio = self.demo_ratio_params[0]
                demo_batch_size = int(self.batch_size * demo_ratio)
                replay_batch_size = self.batch_size - demo_batch_size
                #return replay_batch_size, demo_batch_size
            if self.demo_ratio_type == 'linear_decay':
                t_update = t
                m = 1.0 / self.demo_ratio_params[2]
                demo_ratio = max(self.demo_ratio_params[0] - t_update * m, self.demo_ratio_params[1])
                demo_batch_size = int(self.batch_size * demo_ratio)
                replay_batch_size = self.batch_size - demo_batch_size
                #return replay_batch_size, demo_batch_size

            replay_batch = replay_buffer.sample_batch(replay_batch_size)
            demo_batch = self.demo_buffer.sample_batch(demo_batch_size) 
            return dict(obs=torch.cat((replay_batch['obs'], demo_batch['obs']), 0),
                        obs2=torch.cat((replay_batch['obs2'], demo_batch['obs2']), 0),
                        act=torch.cat((replay_batch['act'], demo_batch['act']), 0),
                        rew=torch.cat((replay_batch['rew'], demo_batch['rew']), 0),
                        done=torch.cat((replay_batch['done'], demo_batch['done']), 0),
                        rew_nstep=torch.cat((replay_batch['rew_nstep'], demo_batch['rew_nstep']), 0),
                        obs_nstep=torch.cat((replay_batch['obs_nstep'], demo_batch['obs_nstep']), 0),                                   
                        done_nstep=torch.cat((replay_batch['done_nstep'], demo_batch['done_nstep']), 0),
                        n_nstep=torch.cat((replay_batch['n_nstep'], demo_batch['n_nstep']), 0)), demo_ratio  

    def start(self,agents,replay_buffer,pause_flag,test2train,sample2train,t_glob,t_limit):

        # Start Training
        epoch = 1
        update_iter = 1
        update_iter_every_log = 0
        total_steps = self.steps_per_epoch * self.epochs
        self.env_error_num = 0
        self.out_of_bounds_num = 0
        self.reward_bonus_num = 0
        checkpoint_test_return = -math.inf

        agent_num = len(agents)
        loss_q_np = np.zeros(agent_num)
        loss_pi_np = np.zeros(agent_num)
        train_ret_np = np.zeros(agent_num)
        train_ret_np = np.zeros(agent_num)
        train_ep_len_np = np.zeros(agent_num)
        train_ep_success_np = np.zeros(agent_num)

        first_update = self.update_every * math.ceil(self.update_after / self.update_every)
        self.save_freq = int(((total_steps - first_update) * self.update_factor_max) / self.num_log_loss_points)
        self.save_freq = max(self.save_freq,1)
        print("Train Logging frequency: " + str(self.save_freq))

        pbar = tqdm(total = int(total_steps + self.update_after), desc = "Training: ", colour="green")
        time0 = time.time()
        pause_flag.value = False
        if self.mode_sync == True: t_limit.value = 0
        t = 0
        while t < total_steps:
            t = replay_buffer.get_t() - self.update_after
            if self.mode_sync == True: t_glob.value = t

            if (t >= 0) and (self.pretrain_bool == True):
                pause_flag.value = True
                for _ in tqdm(range(int(self.pretrain_factor)), desc ="Updating weights (pretraining): ", leave=False):
                    for agent_id in range(agent_num):
                        batch, demo_ratio = self.get_batch(t,replay_buffer)
                        loss_q_np[agent_id], loss_pi_np[agent_id] = agents[agent_id].update(data=batch)
                self.pretrain_bool = False
                if self.mode_sync == True: t_limit.value = update_iter * self.update_every
                pause_flag.value = False

            if (t >= 0) and (t >= update_iter * self.update_every):

                if self.mode_sync:
                    pause_flag.value = True

                update_iter_actual = (t+1) // self.update_every
                if update_iter != update_iter_actual:
                    if update_iter != 1:
                        message = "Update is missed ("+ str(update_iter) + " --> " + str(update_iter_actual) + ") as the sampler is too fast"
                        tqdm.write("[warning]: " + message)  
                        self.logger.print_logfile(message,level = "warning", terminal = False)
                update_iter = update_iter_actual

                for j in tqdm(range(int(self.update_every * self.update_factor_max)), desc ="Updating weights: ", leave=False):
                    update_iter_every_log += 1                   
                    for agent_id in range(agent_num):
                        if j <= (self.update_factor_np[agent_id] * self.update_every):
                            batch, demo_ratio = self.get_batch(t,replay_buffer)
                            loss_q_np[agent_id], loss_pi_np[agent_id] = agents[agent_id].update(data=batch)

                    if update_iter_every_log % self.save_freq == 0:
                        actual_time = time.time() - time0
 
                        for agent_id in range(agent_num):                          
                            train_ret_np[agent_id] = np.mean(self.return_buffer_list[agent_id])
                            train_ep_len_np[agent_id] = np.mean(self.episode_len_buffer_list[agent_id])
                            train_ep_success_np[agent_id] = np.mean(self.ep_success_buffer_list[agent_id])     
                        self.logger.tb_save_train_data_v3(loss_q_np,
                                                          loss_pi_np,
                                                          train_ret_np,
                                                          train_ep_len_np,
                                                          train_ep_success_np,
                                                          self.env_error_num,
                                                          self.out_of_bounds_num,
                                                          self.reward_bonus_num,
                                                          demo_ratio,
                                                          self.heatmap_pick,
                                                          self.heatmap_place,
                                                          t,
                                                          actual_time,
                                                          update_iter_every_log)    

                pause_flag.value = False
                update_iter += 1
                if self.mode_sync == True: t_limit.value = update_iter * self.update_every

            if t >= epoch * self.steps_per_epoch: 

                if agent_num > 1 or self.fallback_safety:
                    pause_flag.value = True

                epoch_real = (t+1) // self.steps_per_epoch
                if epoch != epoch_real:
                    message = "Test is missed at the end of an epoch ("+ str(epoch) + " --> " + str(epoch_real) + ") as the sampler is too fast"
                    tqdm.write("[warning]: " + message)  
                    self.logger.print_logfile(message,level = "warning", terminal = False)   
                epoch = epoch_real    

                model_paths = []
                for agent_id in range(agent_num):
                    model_paths.append(self.logger.get_model_save_path(epoch, agent_id))
                    agents[agent_id].save_model(model_paths[-1])
                
                avg_test_return_np = np.zeros(agent_num)
                avg_test_return_np_bool = np.zeros(agent_num)   
                
                while True:
                    msg = self.handle_messages(test2train,sample2train)
                    if msg is not None:
                        if len(msg) == 4:
                            if msg[0] == "test" and msg[1] == 1 and msg[2] is not None and msg[3] is not None:
                                avg_test_return_np[int(msg[3])] = msg[2]
                                avg_test_return_np_bool[int(msg[3])] = 1.0
                                if np.all(avg_test_return_np_bool == 1.0):
                                    break
                    time.sleep(0.1)
                
                message = "Epoch: " + str(epoch) + " | Checkpoint avg return: " + str(checkpoint_test_return)
                tqdm.write("[info]: " + message)  
                self.logger.print_logfile(message,level = "info", terminal = False) 

                for agent_id in range(agent_num):
                    message = "Epoch: " + str(epoch) + " | Agent " + str(agent_id) +" avg return: " + str(avg_test_return_np[agent_id])
                    tqdm.write("[info]: " + message)  
                    self.logger.print_logfile(message,level = "info", terminal = False) 

                best_agent_id = np.argmax(avg_test_return_np)

                t_log = epoch * self.steps_per_epoch 

                if self.fallback_safety == True:

                    fb_th = self.get_fallback_th(checkpoint_test_return)
                    
                    message = "Epoch: " + str(epoch) + " | Best Agent: " + str(best_agent_id) + " (avg return: " + str(avg_test_return_np[best_agent_id]) + ") Checkpoint avg return: " + str(checkpoint_test_return) + " | Fb th: " + str(fb_th)
                    tqdm.write("[info]: " + message)  
                    self.logger.print_logfile(message,level = "info", terminal = False) 

                    self.logger.tb_writer_add_scalar("test/fallback_diff", avg_test_return_np[best_agent_id] - fb_th, t_log)

                    if avg_test_return_np[best_agent_id] > fb_th:
                        message = "Epoch: " + str(epoch) + " | No Fallback "
                        tqdm.write("[info]: " + message)  
                        self.logger.print_logfile(message,level = "info", terminal = False) 
                        
                        best_model_path = model_paths[best_agent_id]
                        fb_active = 0
                        checkpoint_test_return = avg_test_return_np[best_agent_id]
                    else:
                        message = "Epoch: " + str(epoch) + " | Fallback Safety Activated"
                        tqdm.write("[info]: " + message)  
                        self.logger.print_logfile(message,level = "info", terminal = False) 

                        fb_active = 1
                    
                    self.logger.tb_writer_add_scalar("test/fallback_safety", fb_active, t_log)              
                else:
                    best_model_path = model_paths[best_agent_id]
                    checkpoint_test_return = avg_test_return_np[best_agent_id]
                
                if self.fallback_safety == True or agent_num>1:
                    for agent_id in range(agent_num):
                        agents[agent_id].load_weights(best_model_path,mode="all",eval=False)
                    

                self.logger.tb_writer_add_scalar("test/checkpoint_test_return", checkpoint_test_return, t_log)
                

                ## To be implemented #####################
                #self.logger.save_replay_buffer(replay_buffer, epoch)
                ####################################

                pause_flag.value = False
                epoch += 1
            
            # Handle messages
            self.handle_messages(test2train,sample2train)

            #pbar.update(1)
            pbar.n = t + self.update_after #check this
            pbar.refresh() #check this
        pbar.close()



        

            
               