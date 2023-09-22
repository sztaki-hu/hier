import numpy as np
import torch
import time
import collections
import random

from tqdm import tqdm

from rltrain.envs.builder import make_env
from rltrain.algos.her import HER

CL_TYPES = ['predefined','selfpaced','selfpaceddual','controldiscrete', 'examplebyexample']

class SamplerTrainerTester:

    def __init__(self,device,logger,config,main_args,config_framework):

        self.device = device
        
        self.logger = logger
        self.config = config
        self.config_framework = config_framework

        self.seed = config['general']['seed'] 
        self.agent_type = config['agent']['type']

        self.total_timesteps = int(float(config["trainer"]["total_timesteps"]))
        self.eval_freq=max(int(float(config["eval"]["freq"])), 1)

        self.batch_size = config['trainer']['batch_size'] 
        
        self.update_after = float(config['trainer']['update_after']) 
        self.update_every = int(float(config['trainer']['update_every'])) 

        self.obs_dim = config['environment']['obs_dim']
        self.act_dim = config['environment']['act_dim']
        self.boundary_min = np.array(config['agent']['boundary_min'])[:self.act_dim]
        self.boundary_max = np.array(config['agent']['boundary_max'])[:self.act_dim]

        self.start_steps = float(config['sampler']['start_steps'])
        self.max_ep_len = config['sampler']['max_ep_len']
        self.eval_num_episodes = config['eval']['num_episodes']

        self.model_save_freq = config['logger']['model']['save']['freq']
        self.model_save_best_start_t = config['logger']['model']['save']['best_start_t'] * self.total_timesteps
        self.model_save_mode = config['logger']['model']['save']['mode']

        # Rollout
        self.rollout_stats_window_size = int(config['logger']['rollout']['stats_window_size'])
        self.ep_rew_dq = collections.deque(maxlen=self.rollout_stats_window_size) 
        self.ep_len_dq = collections.deque(maxlen=self.rollout_stats_window_size)
        self.ep_success_dq = collections.deque(maxlen=self.rollout_stats_window_size)

        # Train
        self.train_stats_window_size = int(config['logger']['train']['stats_window_size'])
        self.loss_q_dq = collections.deque(maxlen=self.train_stats_window_size)
        self.loss_pi_dq = collections.deque(maxlen=self.train_stats_window_size)

        # HER
        self.her_goal_selection_strategy = config['buffer']['her']['goal_selection_strategy']
        self.her_active = False if self.her_goal_selection_strategy == "noher" else True
        self.her_n_sampled_goal = config['buffer']['her']['n_sampled_goal'] 

        # CL
        self.cl_mode = config['trainer']['cl']['type']
       
        # Log
        self.print_out_name = '_'.join((self.logger.exp_name,str(self.logger.seed_id)))  

        assert self.cl_mode in CL_TYPES

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

            num_test_episodes (int): Number of episodes to test the deterministic
                policy at the end of each epoch.

            max_ep_len (int): Maximum length of trajectory / episode / rollout.

            logger_kwargs (dict): Keyword args for EpochLogger.

            save_freq (int): How often (in terms of gap between epochs) to save
                the current policy and value function.

        """ 

    def test_agent(self):
        ep_rets = []
        ep_lens = []
        success_num = 0.0
        for j in range(self.eval_num_episodes):
            [o, info], d, ep_ret, ep_len = self.env_eval.reset_with_init_check(), False, 0, 0
            while not(d or (ep_len == self.max_ep_len)):
                # Take deterministic actions at test time 
                if self.agent_type == 'sac':
                    a = self.agent.get_action(o, True)
                elif self.agent_type == 'td3':
                    a = self.agent.get_action(o, 0)
                o, r, terminated, truncated, info = self.env_eval.step(a)
                d = terminated or truncated
                ep_ret += r
                ep_len += 1
            ep_rets.append(ep_ret)
            ep_lens.append(ep_len)
            if info['is_success'] == True: success_num += 1
        
        ep_ret_avg = sum(ep_rets) / len(ep_rets)
        mean_ep_length = sum(ep_lens) / len(ep_lens)
        success_rate = success_num / self.eval_num_episodes
        
        return ep_ret_avg, mean_ep_length, success_rate
            

    def start(self,agent,replay_buffer):

    
        self.env = make_env(self.config, self.config_framework)

        self.env_eval = make_env(self.config, self.config_framework)

        self.HER = HER(self.config,self.env,replay_buffer)

        if self.cl_mode == 'predefined':
            from rltrain.algos.cl_teachers.PredefinedCL import PredefinedCL as CL
        elif self.cl_mode == 'selfpaced':
            from rltrain.algos.cl_teachers.SelfPacedCL import SelfPacedCL as CL
        elif self.cl_mode == 'selfpaceddual':
            from rltrain.algos.cl_teachers.SelfPacedDualCL import SelfPacedDualCL as CL
        elif self.cl_mode == 'controldiscrete':
            from rltrain.algos.cl_teachers.ControlDiscreteCL import ControlDiscreteCL as CL
        elif self.cl_mode == 'examplebyexample':
            from rltrain.algos.cl_teachers.ExampleByExampleCL import ExampleByExampleCL as CL
        else:
            print(self.cl_mode)
            assert False
            return -1   

        self.CL = CL(self.config, self.env, replay_buffer)
        
        self.agent = agent

        init_invalid_num = 0
        reset_num = 0

        o, ep_ret, ep_len = self.CL.reset_env(0), 0, 0

        best_eval_ep_ret = -float('inf')

        print("Training starts: " + self.print_out_name)
        episode = []
        epoch = 0
        time0 = time.time()
        time_start = time0
        t0 = 0

        # Main loop: collect experience in env and update/log each epoch
        for t in tqdm(range(self.total_timesteps), desc ="Training: ", leave=True):
            
            # Until start_steps have elapsed, randomly sample actions
            # from a uniform distribution for better exploration. Afterwards, 
            # use the learned policy. 
            if t > self.start_steps:
                if self.agent_type == 'sac':
                    a = self.agent.get_action(o, False) 
                elif self.agent_type == 'td3':
                    a = self.agent.get_action(o, self.agent.act_noise)
                # a = self.get_action(o)
            else:
                a = self.env.random_sample()
                # a = self.env.env.action_space.sample()

            # Step the env
            o2, r, terminated, truncated, info = self.env.step(a)
            d = terminated or truncated
            ep_ret += r
            ep_len += 1

            # Ignore the "done" signal if it comes from hitting the time
            # horizon (that is, when it's an artificial terminal signal
            # that isn't based on the agent's state)
            d = False if ep_len==self.max_ep_len else d

            # Save transition
            episode.append((o, a, r, o2, d))

            # Super critical, easy to overlook step: make sure to update 
            # most recent observation!
            o = o2

            # End of trajectory handling
            if d or (ep_len == self.max_ep_len):

                ep_succes = 1.0 if info['is_success'] == True else 0.0
                self.ep_success_dq.append(ep_succes)
                if self.CL.store_success_rate: self.CL.cl_ep_success_dq.append(ep_succes)
                
                for (o, a, r, o2, d) in episode:
                    replay_buffer.store(o, a, r, o2, d)
                          
                if self.her_active and truncated:  self.HER.add_virtial_experience(episode)
                    
                episode = []

                # print("-------------------")
                # print(replay_buffer.get_all())
                # assert False

                self.ep_rew_dq.append(ep_ret)
                self.ep_len_dq.append(ep_len)
                o, ep_ret, ep_len = self.CL.reset_env(t), 0, 0

            # Update handling
            if t >= self.update_after and t % self.update_every == 0:
                if replay_buffer.size > 0:
                    for j in range(self.update_every):
                        batch = replay_buffer.sample_batch(self.batch_size)
                        #print(batch)
                        ret_loss_q, ret_loss_pi = self.agent.update(batch, j)
                        self.loss_q_dq.append(ret_loss_q)
                        if ret_loss_pi != None: self.loss_pi_dq.append(ret_loss_pi)

            # End of epoch handling
            if (t+1) % self.eval_freq == 0:

                epoch +=1

                # Test the performance of the deterministic version of the agent.
                eval_mean_reward, eval_mean_ep_length, eval_success_rate = self.test_agent()

                self.logger.tb_writer_add_scalar("eval/mean_reward", eval_mean_reward, t)
                self.logger.tb_writer_add_scalar("eval/mean_ep_length", eval_mean_ep_length, t)
                self.logger.tb_writer_add_scalar("eval/success_rate", eval_success_rate, t)

                best_model_changed = False
                if t > self.model_save_best_start_t:
                    if eval_mean_reward > best_eval_ep_ret:
                        best_eval_ep_ret = eval_mean_reward
                        best_model_changed = True
                
                # Save model 
                if best_model_changed:
                    model_path = self.logger.get_model_save_path(epoch,best_model=True)
                    self.agent.save_model(model_path,self.model_save_mode)
                    
                
                if epoch % self.model_save_freq == 0:
                    model_path = self.logger.get_model_save_path(epoch)
                    self.agent.save_model(model_path,self.model_save_mode)
                
                # Print out
                if (epoch % self.model_save_freq == 0) or best_model_changed:
                    message = self.print_out_name +  " | t: " + str(t) +  " | epoch: " + str(epoch) + " | eval_mean_reward: " + str(eval_mean_reward) + " | eval_mean_ep_length: " + str(eval_mean_ep_length) + " | eval_success_rate: " + str(eval_success_rate) + " | cl_ratio: " + str(self.CL.cl_ratio)
                    if best_model_changed: message += " *" 
                    tqdm.write("[info] " + message)     

                # ROLLOUT
                self.logger.tb_writer_add_scalar("rollout/ep_rew_mean", np.mean(self.ep_rew_dq), t)
                self.logger.tb_writer_add_scalar("rollout/ep_len_mean", np.mean(self.ep_len_dq), t)
                self.logger.tb_writer_add_scalar("rollout/success_rate", np.mean(self.ep_success_dq), t)

                # TRAIN
                self.logger.tb_writer_add_scalar('train/critic_loss', np.mean(self.loss_q_dq), t)
                self.logger.tb_writer_add_scalar("train/actor_loss", np.mean(self.loss_pi_dq), t)

                self.logger.tb_writer_add_scalar("cl/ratio", self.CL.cl_ratio, t)

                # invalid_init_ratio = float(init_invalid_num) / reset_num 
                # self.logger.tb_writer_add_scalar("train/invalid_init_ratio", invalid_init_ratio, t)

                # TIME
                time1 = time.time()
                time_delta = time1 - time0
                fps =  (t-t0) / time_delta

                self.logger.tb_writer_add_scalar('time/fps', fps, t)
                self.logger.tb_writer_add_scalar('time/all', time1 - time_start, t)

                time0 = time1
                t0 = t

        self.logger.tb_close()
   
    