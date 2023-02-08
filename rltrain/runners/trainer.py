import os
import math
import numpy as np
import torch
import time

from tqdm import tqdm

from multiprocessing import Process

from rltrain.buffers.replay import ReplayBuffer
from rltrain.agents.agent import Agent
from rltrain.runners.tester import Tester

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
        self.update_factor = config['trainer']['update_factor'] 

        self.num_log_loss_points = config['logger']['num_log_loss_points'] 

        self.obs_dim = config['environment']['obs_dim']
        self.act_dim = config['environment']['act_dim']
        self.boundary_min = np.array(config['agent']['boundary_min'])[:self.act_dim]
        self.boundary_max = np.array(config['agent']['boundary_max'])[:self.act_dim]

        self.demo_use = config['demo']['demo_use']  
        self.demo_ratio = config['demo']['demo_ratio']

        if self.demo_use:
            self.demo_batch_size = int(self.batch_size * self.demo_ratio)
            self.replay_batch_size = self.batch_size - self.demo_batch_size
        
        self.processes = []

        

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

    
    def start(self,agent,replay_buffer,pause_flag,env_error_num,test2train,sample2train):

        print(agent)

        # Start Training
        t = 0
        ptr_old = 0
        epoch = 1
        update_iter = 1
        update_iter_every_log = 0
        total_steps = self.steps_per_epoch * self.epochs

        first_update = self.update_every * math.ceil(self.update_after / self.update_every)
        self.save_freq = int(((total_steps - first_update) * self.update_factor) / self.num_log_loss_points)
        self.save_freq = max(self.save_freq,1)


        pbar = tqdm(total = total_steps, desc = "Training: ", colour="green")
        time0 = time.time()
        pause_flag.value = False
        while t < total_steps:
            ptr_now = replay_buffer.get_ptr()
            diff = ptr_now - ptr_old
            if diff >= 0:
                t += diff
            else:
                t += self.replay_size - ptr_old + ptr_now
            ptr_old = ptr_now
            # tqdm.write("t " + str(t))
            # tqdm.write("ptr_now " + str(ptr_now))

            if (t > self.update_after) and (t >= update_iter * self.update_every):
                #tqdm.write("UPDATE " + str(update_iter))

                if self.mode_sync:
                    pause_flag.value = True

                update_iter_actual = (t+1) // self.update_every
                if update_iter != update_iter_actual:
                    if update_iter != 1:
                        tqdm.write("! Update is missed ("+ str(update_iter) + " --> " + str(update_iter_actual) + ") as the sampler is too fast")   
                update_iter = update_iter_actual


                if self.demo_use == False:
                    for j in tqdm(range(int(self.update_every * self.update_factor)), desc ="Updating weights: ", leave=False):
                        update_iter_every_log += 1
                        batch = replay_buffer.sample_batch(self.batch_size)    
                        #tqdm.write(str(batch))
                        loss_q, loss_pi = agent.update(data=batch)
                        if update_iter_every_log % self.save_freq == 0:
                            env_error_num_value = env_error_num.value
                            actual_time = time.time() - time0
                            self.logger.tb_save_train_data_v2(loss_q,loss_pi,env_error_num_value,t,actual_time,update_iter_every_log)
                else:
                    for j in tqdm(range(int(self.update_every * self.update_factor)), desc ="Updating weights: ", leave=False):
                        update_iter_every_log += 1
                        replay_batch = replay_buffer.sample_batch(self.replay_batch_size)
                        demo_batch = self.demo_buffer.sample_batch(self.demo_batch_size) 
                        batch = dict(obs=torch.cat((replay_batch['obs'], demo_batch['obs']), 0),
                                    obs2=torch.cat((replay_batch['obs2'], demo_batch['obs2']), 0),
                                    act=torch.cat((replay_batch['act'], demo_batch['act']), 0),
                                    rew=torch.cat((replay_batch['rew'], demo_batch['rew']), 0),
                                    done=torch.cat((replay_batch['done'], demo_batch['done']), 0))  
                        loss_q, loss_pi = agent.update(data=batch)
                        if update_iter_every_log % self.save_freq == 0:
                            env_error_num_value = env_error_num.value
                            actual_time = time.time() - time0
                            self.logger.tb_save_train_data_v2(loss_q,loss_pi,env_error_num_value,t,actual_time,update_iter_every_log)    

                pause_flag.value = False
                update_iter += 1    

            if t >= epoch * self.steps_per_epoch: 

                epoch_real = (t+1) // self.steps_per_epoch
                if epoch != epoch_real:
                    tqdm.write("! Test is missed at the end of an epoch ("+ str(epoch) + " --> " + str(epoch_real) + ") as the sampler is too fast")   
                epoch = epoch_real  

                pause_flag.value = True

                agent.save_model(self.logger.get_model_save_path(epoch))

                pause_flag.value = False

                epoch += 1

            if test2train.empty() == False:
                data = test2train.get()
                if data['code'] < 0:
                    tqdm.write("Error Code: " + str(data['code']) + " Description: " + str(data['description']))
                else:
                    if data['code'] == 1:
                        avg_test_return = data['value']
                        epoch_test = data['epoch']
                          
                        test_t = epoch_test * self.steps_per_epoch           
                        log_text = "AVG test return: " + str(epoch_test) + ". epoch ("+ str(test_t) + " transitions) : " + str(avg_test_return)
                        tqdm.write(log_text) 
                
            
            if sample2train.empty() == False:
                data = sample2train.get()
                if data['code'] < 0:                 
                    tqdm.write("Error Code: " + str(data['code']) + " Description: " + str(data['description']))

            #pbar.update(1)
            pbar.n = t #check this
            pbar.refresh() #check this
        pbar.close()

        # for p in self.processes:
        #     p.join()
            
               