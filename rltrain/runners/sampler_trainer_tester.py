import os
import math
import numpy as np
import torch

from tqdm import tqdm

from rltrain.buffers.replay import ReplayBuffer
from rltrain.agents.agent import Agent

class SamplerTrainerTester:

    def __init__(self, device,env,demo_buffer,logger,config):
        self.device = device
        self.env = env
        self.demo_buffer = demo_buffer
        self.logger = logger

        self.seed = config['general']['seed'] 
        self.steps_per_epoch = config['trainer']['steps_per_epoch'] 
        self.epochs = config['trainer']['epochs'] 

        self.batch_size = config['trainer']['batch_size'] 
        self.start_steps = config['sampler']['start_steps'] 
        self.update_after = config['trainer']['update_after'] 
        self.update_every = config['trainer']['update_every'] 
        self.update_factor = config['trainer']['update_factor'] 
        self.num_test_episodes = config['tester']['num_test_episodes'] 
        self.max_ep_len = config['sampler']['max_ep_len'] 
        self.num_log_loss_points = config['logger']['num_log_loss_points'] 

        self.obs_dim = config['environment']['obs_dim']
        self.act_dim = config['environment']['act_dim']
        self.boundary_min = np.array(config['agent']['boundary_min'])[:self.act_dim]
        self.boundary_max = np.array(config['agent']['boundary_max'])[:self.act_dim]

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self.max_avg_test_return = - 100000

        self.replay_size = int(config['buffer']['replay_buffer_size']) 
        self.replay_buffer = ReplayBuffer(obs_dim=self.obs_dim, act_dim=self.act_dim, size=self.replay_size)

        self.demo_use = config['demo']['demo_use']  
        self.demo_ratio = config['demo']['demo_ratio']

        if self.demo_use:
            self.demo_batch_size = int(self.batch_size * self.demo_ratio)
            self.replay_batch_size = self.batch_size - self.demo_batch_size

        self.agent = Agent(device,config)

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
    
    def create_demos(self):
        if self.demo_buffer is not None:
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
            self.logger.save_demos(self.demo_name, self.demo_buffer)
        # batch = self.demo_buffer.sample_batch(self.batch_size) 
        # print(batch)
        # print(self.demo_buffer.ptr)
    
    def load_demos(self):
        self.demo_buffer = self.logger.load_demos(self.demo_name)
        # batch = self.demo_buffer.sample_batch(self.batch_size) 
        # print(batch)

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

        avg_return = self.test_agent(verbose = True)

        print("########################################")
        print("avg return: " + str(avg_return))
    
    def test_agent(self, epoch = None, verbose = False):
        sum_return = 0
        for j in tqdm(range(self.num_test_episodes), desc ="Testing: ", leave=False):
            o, d, ep_ret, ep_len = self.env.reset(), False, 0, 0
            while not(d or (ep_len == self.max_ep_len)):
                # Take deterministic actions at test time 
                try:
                    a = self.agent.get_action(o, True)
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
        return avg_return
    
    def sample_batch(self):
        if self.demo_use == False:
            batch = self.replay_buffer.sample_batch(self.batch_size)    
        else:
            replay_batch = self.replay_buffer.sample_batch(self.replay_batch_size)
            demo_batch = self.demo_buffer.sample_batch(self.demo_batch_size) 
            batch = dict(obs=torch.cat((replay_batch['obs'], demo_batch['obs']), 0),
                        obs2=torch.cat((replay_batch['obs2'], demo_batch['obs2']), 0),
                        act=torch.cat((replay_batch['act'], demo_batch['act']), 0),
                        rew=torch.cat((replay_batch['rew'], demo_batch['rew']), 0),
                        done=torch.cat((replay_batch['done'], demo_batch['done']), 0))
        return batch

    def start(self):
        
        # Prepare for interaction with environment
        total_steps = self.steps_per_epoch * self.epochs
        first_update = self.update_every * math.ceil(self.update_after / self.update_every)
        self.save_freq = int(((total_steps - first_update) * self.update_factor) / self.num_log_loss_points)
        self.save_freq = max(self.save_freq,1)
    
        o, ep_ret, ep_len = self.env.reset(), 0, 0

        # Main loop: collect experience in env and update/log each epoch
        episode_iter = 0
        env_error_num = 0
        sum_ep_len = 0
        sum_ep_ret = 0
        update_iter = 0
        log_loss_iter = 1

        epoch = 0
        avg_test_return = self.test_agent(epoch)
        log_text = "AVG test return: " + str(epoch) + ". epoch : " + str(avg_test_return)
        print(log_text) 
        self.logger.tb_writer_add_scalar("test/average_return", avg_test_return, epoch)

        pbar = tqdm(total = total_steps, desc ="Training: ", colour="green")
        t = 0
        while t < total_steps:
            
            #tqdm.write(str(t)) 
            # tqdm.write("o: " + str(o))

            # Until start_steps have elapsed, randomly sample actions
            # from a uniform distribution for better exploration. Afterwards, 
            # use the learned policy. 
            if t > self.start_steps:
                a = self.agent.get_action(o)
            else:
                a = self.agent.get_random_action()

            # Step the env
            #tqdm.write("Action: "+str(a))
            try:
                o2, r, d, info = self.env.step(a)
            except:
                tqdm.write("Error in simulation, thus reseting the environment")
                tqdm.write("a: " + str(a)) 
                env_error_num += 1
                o, ep_ret, ep_len = self.env.reset(), 0, 0   
                continue

            # tqdm.write("o2: " + str(o2)) 
            # tqdm.write("r: " + str(r)) 
            # tqdm.write("d: " + str(d)) 
        
            ep_ret += r
            ep_len += 1

            # Ignore the "done" signal if it comes from hitting the time
            # horizon (that is, when it's an artificial terminal signal
            # that isn't based on the agent's state)
            d = False if ep_len==self.max_ep_len else d

            # Store experience to replay buffer
            self.replay_buffer.store(o, a, r, o2, d)

            # Super critical, easy to overlook step: make sure to update 
            # most recent observation!
            o = o2

            # End of trajectory handling
            if d or (ep_len == self.max_ep_len):               
                episode_iter += 1
                sum_ep_len += sum_ep_len
                sum_ep_ret += ep_ret
                o, ep_ret, ep_len = self.env.reset(), 0, 0
                

            # Update handling
            if t >= self.update_after and t % self.update_every == 0:
                for j in tqdm(range(self.update_every * self.update_factor), desc ="Updating weights: ", leave=False):
                    update_iter += 1
                    batch = self.sample_batch()
                    loss_q, loss_pi = self.agent.update(data=batch)
                    if update_iter % self.save_freq == 0:
                        self.logger.tb_save_train_data(loss_q,loss_pi,sum_ep_len,sum_ep_ret,episode_iter,env_error_num,t,log_loss_iter)
                        sum_ep_len = 0
                        sum_ep_ret = 0
                        episode_iter = 0
                        log_loss_iter += 1

            # End of epoch handling
            if (t+1) % self.steps_per_epoch == 0:
                epoch = (t+1) // self.steps_per_epoch                
                
                avg_test_return = self.test_agent(epoch)
                log_text = "AVG test return: " + str(epoch) + ". epoch ("+ str(t+1) + " transitions) : " + str(avg_test_return)
                tqdm.write(log_text) 
                self.logger.tb_writer_add_scalar("test/average_return", avg_test_return, epoch)
                
            

                if avg_test_return > self.max_avg_test_return: 
                    self.max_avg_test_return = avg_test_return
                    self.logger.save_model(self.agent.ac.pi.state_dict(),epoch)
            
            t += 1           
            pbar.update(1)   