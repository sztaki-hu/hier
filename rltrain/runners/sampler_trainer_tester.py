import numpy as np
import torch
import time
import collections
import random

from tqdm import tqdm

from rltrain.envs.builder import make_env
from rltrain.algos.cl.builder import make_cl
from rltrain.agents.builder import make_agent
from rltrain.buffers.builder import make_per
from rltrain.algos.highlights.builder import make_hl
from rltrain.algos.her import HER

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
        self.model_save_measure = config['logger']['model']['save']['measure']
        assert self.model_save_measure in ['reward','success_rate']

        # Rollout
        self.rollout_stats_window_size = int(config['logger']['rollout']['stats_window_size'])
        self.ep_rew_dq = collections.deque(maxlen=self.rollout_stats_window_size) 
        self.ep_len_dq = collections.deque(maxlen=self.rollout_stats_window_size)
        self.ep_success_dq = collections.deque(maxlen=self.rollout_stats_window_size)

        # State change
        self.state_change_stats_window_size = int(config['logger']['state_change']['stats_window_size'])
        self.ep_state_changed_dq = collections.deque(maxlen=self.state_change_stats_window_size)

        # Train
        self.train_stats_window_size = int(config['logger']['train']['stats_window_size'])
        self.loss_q_dq = collections.deque(maxlen=self.train_stats_window_size)
        self.loss_pi_dq = collections.deque(maxlen=self.train_stats_window_size)

        # HER
        self.her_goal_selection_strategy = config['buffer']['her']['goal_selection_strategy']
        self.her_active = False if self.her_goal_selection_strategy == "noher" else True
        self.her_n_sampled_goal = config['buffer']['her']['n_sampled_goal'] 
        self.virtual_experience_dq = collections.deque(maxlen=self.rollout_stats_window_size)     

        # Log
        self.print_out_name = '_'.join((self.logger.exp_name,str(self.logger.seed_id)))  

        # Replay Buffer / PER (Prioritized Experience Replay)
        self.per_active = False if config['buffer']['per']['mode'] == 'noper' else True
        self.replay_buffer = make_per(config)
        
        # Agent
        self.agent = make_agent(device,config,config_framework)

        # Env train
        self.env = make_env(self.config, self.config_framework)

        # Env eval
        self.env_eval = make_env(self.config, self.config_framework)

        # HER (Hindsight Experience Replay)
        self.HER = HER(self.config,self.env,self.replay_buffer)

        # CL (Curriculum Learning)
        self.cl_mode = config['trainer']['cl']['type']
        self.CL = make_cl(self.config, self.env, self.replay_buffer)

        # HL (Highlights)
        self.highlights_include_test =  config['buffer']['highlights']['include_test']

        self.HL = make_hl(self.config)
        if self.HL.hl_active == False: self.replay_batch_size = self.batch_size
        

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

    def test_agent(self,train_t):
        ep_rets = []
        ep_lens = []
        success_num = 0.0
        state_changed_num = 0
        
        for j in range(self.eval_num_episodes):
            o, d, ep_ret, ep_len = self.env_eval.reset(), False, 0, 0
            o_init = o.copy()
            self.env_eval.ep_o_start = o.copy()
            test_episode = []
            while not(d or (ep_len == self.max_ep_len)):
                # Take deterministic actions at test time 
                if self.agent_type == 'sac':
                    a = self.agent.get_action(o, True)
                elif self.agent_type in ['td3','ddpg']:
                    a = self.agent.get_action(o, 0)
                o2, r, terminated, truncated, info = self.env_eval.step(a)
                d = terminated or truncated
                # Save transition
                test_episode.append((o, a, r, o2, d))
                o = o2
                ep_ret += r
                ep_len += 1    
            if self.highlights_include_test: self.HL.store_episode(test_episode,info['is_success'],train_t)               
            ep_rets.append(ep_ret)
            ep_lens.append(ep_len)
            if info['is_success'] == True: success_num += 1
            if self.env_eval.is_diff_state(o_init,o,threshold = 0.01): state_changed_num += 1   
        
        ep_ret_avg = sum(ep_rets) / len(ep_rets)
        mean_ep_length = sum(ep_lens) / len(ep_lens)
        success_rate = success_num / self.eval_num_episodes
        state_change_rate = state_changed_num / self.eval_num_episodes
        
        return ep_ret_avg, mean_ep_length, success_rate,state_change_rate
            

    def start(self):

        # Env reset
        o, ep_ret, ep_len = self.CL.reset_env(0), 0, 0
        self.env.ep_o_start = o.copy()

        # Init variables
        best_eval_measure = -float('inf')
        episode = []
        epoch = 0
        time0 = time.time()
        time_start = time0
        t0 = 0
        t_collect = 0
        t_process_ep = 0
        t_train = 0
        t_test = 0

        print("Training starts: " + self.print_out_name)

        # Main loop: collect experience in env and update/log each epoch
        for t in tqdm(range(self.total_timesteps), desc ="Training: ", leave=True):
            
            t_collect_0 = time.time()
            # Until start_steps have elapsed, randomly sample actions
            # from a uniform distribution for better exploration. Afterwards, 
            # use the learned policy. 
            if t > self.start_steps:
                if self.agent_type == 'sac':
                    a = self.agent.get_action(o, False) 
                elif self.agent_type in ['td3','ddpg']:
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

            t_collect += (time.time() - t_collect_0)

            
            # End of trajectory handling
            if d or (ep_len == self.max_ep_len):
                t_process_ep_0 = time.time()

                ep_succes = 1.0 if info['is_success'] == True else 0.0
                self.ep_success_dq.append(ep_succes)
                if self.CL.store_rollout_success_rate: self.CL.cl_rollout_success_dq.append(ep_succes)
                
                for (o, a, r, o2, d) in episode:
                    self.replay_buffer.store(o, a, r, o2, d)
                
                # Highlights
                self.HL.store_episode(episode,info['is_success'],t)

                # HER
                if self.her_active and truncated: 
                    virtual_experience_added = self.HER.add_virtial_experience(episode)
                    self.virtual_experience_dq.append(virtual_experience_added)

                self.ep_state_changed_dq.append(1.0) if self.env.is_diff_state(episode[0][0],episode[-1][0],threshold = 0.01) else self.ep_state_changed_dq.append(0.0)

                episode = []
                self.ep_rew_dq.append(ep_ret)
                self.ep_len_dq.append(ep_len)

                # CL 
                o, ep_ret, ep_len = self.CL.reset_env(t), 0, 0
                self.env.ep_o_start = o.copy()

                t_process_ep += (time.time() - t_process_ep_0)

            
            # Update handling
            if t >= self.update_after and t % self.update_every == 0:
                t_train_0 = time.time()
                if self.replay_buffer.size > 0:
                    for j in range(self.update_every):  
                        # SAMPLE BATCH         
                        if self.HL.hl_active and self.HL.is_sampling_possible():
                            self.replay_batch_size = self.HL.get_replay_batch_size()
                            replay_batch = self.replay_buffer.sample_batch(self.replay_batch_size)

                            highlights_batch = self.HL.sample_batch()
                        
                            batch = dict(obs=torch.cat((replay_batch['obs'], highlights_batch['obs']), 0),
                                            obs2=torch.cat((replay_batch['obs2'], highlights_batch['obs2']), 0),
                                            act=torch.cat((replay_batch['act'], highlights_batch['act']), 0),
                                            rew=torch.cat((replay_batch['rew'], highlights_batch['rew']), 0),
                                            done=torch.cat((replay_batch['done'], highlights_batch['done']), 0),
                                            indices=torch.cat((replay_batch['indices'], highlights_batch['indices']), 0),
                                            weights=torch.cat((replay_batch['weights'], highlights_batch['weights']), 0))
                        else:     
                            batch = self.replay_buffer.sample_batch(self.batch_size)  

                        # UPDATE WEIGHTS
                        ret_loss_q, ret_loss_pi, batch_priorities = self.agent.update(batch, j)

                        # PER
                        if self.per_active: 
                            batch_indices = batch['indices'].detach().cpu().numpy().astype(int)
                            self.replay_buffer.update_priorities(batch_indices[:self.replay_batch_size], batch_priorities[:self.replay_batch_size])
                        
                        # PHiER
                        if self.HL.hl_active and self.HL.is_sampling_possible():
                            self.HL.update_priority(batch_priorities,self.replay_batch_size)
                        
                        self.loss_q_dq.append(ret_loss_q)
                        if ret_loss_pi != None: self.loss_pi_dq.append(ret_loss_pi)
                t_train += (time.time() - t_train_0)

            

            # End of epoch handling
            if (t+1) % self.eval_freq == 0:
                t_test_0 = time.time()

                epoch +=1

                # Test the performance of the deterministic version of the agent.
                eval_mean_reward, eval_mean_ep_length, eval_success_rate, eval_state_change_rate = self.test_agent(t)

                if self.CL.store_eval_success_rate: self.CL.cl_eval_success_dq.append(eval_success_rate)

                self.logger.tb_writer_add_scalar("eval/mean_reward", eval_mean_reward, t)
                self.logger.tb_writer_add_scalar("eval/mean_ep_length", eval_mean_ep_length, t)
                self.logger.tb_writer_add_scalar("eval/success_rate", eval_success_rate, t)
                self.logger.tb_writer_add_scalar("eval/state_change_rate", eval_state_change_rate, t)

                best_model_changed = False
                if self.model_save_measure == 'reward':
                    eval_measure = eval_mean_reward 
                elif self.model_save_measure == 'success_rate':
                    eval_measure = eval_success_rate
                if t > self.model_save_best_start_t:
                    if eval_measure > best_eval_measure:
                        best_eval_measure = eval_measure
                        best_model_changed = True
                
                # Save model 
                if best_model_changed:
                    model_path = self.logger.get_model_save_path("best_model")
                    self.agent.save_model(model_path,"all")
                    
                
                if epoch % self.model_save_freq == 0:
                    model_path = self.logger.get_model_save_path(epoch)
                    self.agent.save_model(model_path,self.model_save_mode)
                
                # Print out
                if (epoch % self.model_save_freq == 0) or best_model_changed:              
                    message = " | ".join([self.print_out_name,
                                    "t: " + str(t),
                                    "epoch: " + str(epoch),
                                    "eval_mean_reward " + str(eval_mean_reward),
                                    "eval_mean_ep_length: " + str(eval_mean_ep_length),
                                    "eval_success_rate: " + str(eval_success_rate),
                                    "ratios [CL]: " + str([round(self.CL.cl_ratio, 2)])])

                    if best_model_changed: message += " *" 
                    tqdm.write("[info] " + message)     

                # ROLLOUT
                self.logger.tb_writer_add_scalar("rollout/ep_rew_mean", np.mean(self.ep_rew_dq), t)
                self.logger.tb_writer_add_scalar("rollout/ep_len_mean", np.mean(self.ep_len_dq), t)
                self.logger.tb_writer_add_scalar("rollout/success_rate", np.mean(self.ep_success_dq), t)

                # STATE CHANGED
                self.logger.tb_writer_add_scalar("rollout/state_changed", np.mean(self.ep_state_changed_dq), t)

                # TRAIN
                self.logger.tb_writer_add_scalar('train/critic_loss', np.mean(self.loss_q_dq), t)
                self.logger.tb_writer_add_scalar("train/actor_loss", np.mean(self.loss_pi_dq), t)

                # HER
                self.logger.tb_writer_add_scalar("her/virtual_experience_added", np.mean(self.virtual_experience_dq), t)
                
                # PER
                if self.per_active:
                    self.logger.tb_writer_add_scalar("per/beta", np.mean(self.replay_buffer.get_beta()), t)

                # CL
                self.logger.tb_writer_add_scalar("cl/ratio", self.CL.cl_ratio, t)
                if self.cl_mode == 'examplebyexample': self.logger.tb_writer_add_scalar("cl/same_setup_num", np.mean(self.CL.same_setup_num_dq), t)

                # HL
                if self.HL.hl_mode != 'multifix':
                    self.logger.tb_writer_add_scalar("hl/highlights_buffer_size", self.HL.hl_replay_buffer.size, t)
                    self.logger.tb_writer_add_scalar("hl/highlights_threshold", self.HL.hl_threshold, t)
                    self.logger.tb_writer_add_scalar("hl/highlights_batch_ratio", self.HL.hl_batch_ratio, t)
                    self.logger.tb_writer_add_scalar("hl/highlights_batch_size", self.HL.hl_batch_size, t)
                else:
                    for hl_index in range(self.HL.hl_bin_num): 
                        self.logger.tb_writer_add_scalar("hl/highlights_buffer_size_"+str(hl_index), self.HL.hl_replay_buffers[hl_index].size, t)
                        self.logger.tb_writer_add_scalar("hl/highlights_threshold_"+str(hl_index), self.HL.hl_thresholds[hl_index], t)
                        self.logger.tb_writer_add_scalar("hl/highlights_batch_ratio_"+str(hl_index), self.HL.hl_batch_ratios[hl_index], t)       
                        self.logger.tb_writer_add_scalar("hl/highlights_batch_size_"+str(hl_index), self.HL.hl_batch_sizes[hl_index], t)      

                # TIME
                time1 = time.time()
                time_delta = time1 - time0
                fps =  (t-t0) / time_delta
                t_test = time.time() - t_test_0

                # General
                self.logger.tb_writer_add_scalar('time/fps', fps, t)
                self.logger.tb_writer_add_scalar('time/total', time1 - time_start, t)
                # Abs
                self.logger.tb_writer_add_scalar('time/abs_collect', t_collect, t)
                self.logger.tb_writer_add_scalar('time/abs_process_ep', t_process_ep, t)
                self.logger.tb_writer_add_scalar('time/abs_train', t_train, t)
                self.logger.tb_writer_add_scalar('time/abs_test', t_test, t)
                self.logger.tb_writer_add_scalar('time/abs_other', (time_delta - t_collect - t_process_ep - t_train - t_test), t)
                self.logger.tb_writer_add_scalar('time/abs_all', time_delta, t)
                # Share
                self.logger.tb_writer_add_scalar('time/share_collect', t_collect / time_delta, t)
                self.logger.tb_writer_add_scalar('time/share_process_ep', t_process_ep / time_delta, t)
                self.logger.tb_writer_add_scalar('time/share_train', t_train / time_delta, t)
                self.logger.tb_writer_add_scalar('time/share_test', t_test / time_delta, t)
                self.logger.tb_writer_add_scalar('time/share_other', (time_delta - t_collect - t_process_ep - t_train - t_test) / time_delta, t)

                time0 = time1
                t0 = t
                
                t_collect = 0
                t_process_ep = 0
                t_train = 0

        self.logger.tb_close()
   
    