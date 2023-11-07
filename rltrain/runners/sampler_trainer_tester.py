import numpy as np
import torch
import time
import collections
import random

from tqdm import tqdm

from rltrain.taskenvs.builder import make_taskenv
from rltrain.algos.cl.builder import make_cl
from rltrain.agents.builder import make_agent
from rltrain.buffers.builder import make_per
from rltrain.algos.hier.builder import make_hier
from rltrain.algos.her.HER import HER
from rltrain.utils.utils import safe_dq_mean

class SamplerTrainerTester:

    def __init__(self,device,logger,config,config_framework):

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
        self.virtual_experience_dq = collections.deque(maxlen=self.rollout_stats_window_size)     

        # Log
        self.print_out_name = '_'.join((self.logger.exp_name,str(self.logger.seed_id)))  

        # Replay Buffer / PER (Prioritized Experience Replay)
        self.per_active = False if config['buffer']['per']['mode'] == 'noper' else True
        self.replay_buffer = make_per(self.config, self.config_framework)
        
        # Agent
        self.agent = make_agent(device, self.config, self.config_framework)

        # Env train
        self.taskenv = make_taskenv(self.config, self.config_framework)

        # Env eval
        self.taskenv_eval = make_taskenv(self.config, self.config_framework)

        # HER (Hindsight Experience Replay)
        self.HER = HER(self.config, self.config_framework, self.taskenv, self.replay_buffer)

        # CL (Curriculum Learning)
        self.CL = make_cl(self.config, self.config_framework, self.taskenv)

        # HiER
        self.hier_include_test =  config['buffer']['hier']['include_test']
        self.hier_include_train =  config['buffer']['hier']['include_train']
    
        self.HiER = make_hier(self.config, self.config_framework)
        if self.HiER.active == False: self.ser_batch_size = self.batch_size
        

    def test_agent(self,train_t):
        ep_rets = []
        ep_lens = []
        success_num = 0.0
        state_changed_num = 0
        
        for j in range(self.eval_num_episodes):
            o, d, ep_ret, ep_len = self.taskenv_eval.reset(), False, 0, 0
            o_init = o.copy()
            test_episode = []
            info = {}
            info['is_success'] = False
            while not(d or (ep_len == self.max_ep_len)):
                # Take deterministic actions at test time 
                a = self.agent.get_action(o = o, deterministic = True, noise_scale = 0.0)
                o2, r, terminated, truncated, info = self.taskenv_eval.step(a)
                d = terminated or truncated
                # Save transition
                test_episode.append((o, a, r, o2, d))
                o = o2
                ep_ret += r
                ep_len += 1    
            if self.hier_include_test: self.HiER.store_episode(test_episode, info['is_success'], train_t)               
            ep_rets.append(ep_ret)
            ep_lens.append(ep_len)
            if info['is_success'] == True: success_num += 1
            if self.taskenv_eval.is_diff_state(o_init,o,threshold = 0.01): state_changed_num += 1   
        
        ep_ret_avg = sum(ep_rets) / len(ep_rets)
        mean_ep_length = sum(ep_lens) / len(ep_lens)
        success_rate = success_num / self.eval_num_episodes
        state_change_rate = state_changed_num / self.eval_num_episodes
        
        return ep_ret_avg, mean_ep_length, success_rate,state_change_rate
            

    def start(self):

        # Env reset
        o, ep_ret, ep_len = self.CL.reset_env(0), 0, 0

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
        info = {}
        info['is_success'] = False

        print("Training starts: " + self.print_out_name)

        # Main loop: collect experience in env and update/log each epoch
        for t in tqdm(range(self.total_timesteps), desc ="Training: ", leave=True):
            
            t_collect_0 = time.time()
            # Until start_steps have elapsed, randomly sample actions
            # from a uniform distribution for better exploration. Afterwards, 
            # use the learned policy. 
            if t > self.start_steps:
                a = self.agent.get_action(o = o, deterministic = False, noise_scale = self.agent.act_noise)
            else:
                a = self.taskenv.random_sample()

            # Step the env
            o2, r, terminated, truncated, info = self.taskenv.step(a)
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
                self.CL.append_rollout_success_dq(ep_succes)
                #if self.CL.store_rollout_success_rate: self.CL.rollout_success_dq.append(ep_succes)
                
                for (o, a, r, o2, d) in episode:
                    self.replay_buffer.store(o, a, r, o2, d)
                
                # Hier
                if self.hier_include_train: self.HiER.store_episode(episode,info['is_success'],t)

                # HER
                if self.her_active and truncated: 
                    virtual_experience_added = self.HER.add_virtial_experience(episode)
                    self.virtual_experience_dq.append(virtual_experience_added)

                self.ep_state_changed_dq.append(1.0) if self.taskenv.is_diff_state(episode[0][0],episode[-1][0],threshold = 0.01) else self.ep_state_changed_dq.append(0.0)

                episode = []
                self.ep_rew_dq.append(ep_ret)
                self.ep_len_dq.append(ep_len)

                # CL 
                o, ep_ret, ep_len = self.CL.reset_env(t), 0, 0

                t_process_ep += (time.time() - t_process_ep_0)

            
            # Update handling
            if t >= self.update_after and t % self.update_every == 0:
                t_train_0 = time.time()
                if self.replay_buffer.size > 0:
                    for j in range(self.update_every):  
                        # SAMPLE BATCH    
                         
                        if self.HiER.active and self.HiER.is_sampling_possible():   
                            self.ser_batch_size = self.HiER.get_ser_batch_size()        
                            ser_batch = self.replay_buffer.sample_batch(self.ser_batch_size)
                            hier_batch = self.HiER.sample_batch()
                        
                            batch = dict(obs=torch.cat((ser_batch['obs'], hier_batch['obs']), 0),
                                            obs2=torch.cat((ser_batch['obs2'], hier_batch['obs2']), 0),
                                            act=torch.cat((ser_batch['act'], hier_batch['act']), 0),
                                            rew=torch.cat((ser_batch['rew'], hier_batch['rew']), 0),
                                            done=torch.cat((ser_batch['done'], hier_batch['done']), 0),
                                            indices=torch.cat((ser_batch['indices'], hier_batch['indices']), 0),
                                            weights=torch.cat((ser_batch['weights'], hier_batch['weights']), 0))
                        else:     
                            batch = self.replay_buffer.sample_batch(self.batch_size)  
                            self.ser_batch_size = self.batch_size

                        # UPDATE WEIGHTS
                        ret_loss_q, ret_loss_pi, batch_priorities = self.agent.update(batch, j)

                        # PER
                        if self.per_active: 
                            batch_indices = batch['indices'].detach().cpu().numpy().astype(int)
                            self.replay_buffer.update_priorities(batch_indices[:self.ser_batch_size], batch_priorities[:self.ser_batch_size])
                        
                        # PHiER
                        if self.HiER.active and self.HiER.is_sampling_possible():
                            self.HiER.update_priority(batch_priorities,self.ser_batch_size)
                        
                        self.loss_q_dq.append(ret_loss_q)
                        if ret_loss_pi != None: self.loss_pi_dq.append(ret_loss_pi)
                t_train += (time.time() - t_train_0)

            

            # End of epoch handling
            if (t+1) % self.eval_freq == 0:
                t_test_0 = time.time()

                epoch +=1

                # Test the performance of the deterministic version of the agent.
                eval_mean_reward, eval_mean_ep_length, eval_success_rate, eval_state_change_rate = self.test_agent(t)

                self.CL.append_eval_success_dq(eval_success_rate)
                #if self.CL.store_eval_success_rate: self.CL.cl_eval_success_dq.append(eval_success_rate)

                self.logger.tb_writer_add_scalar("eval/mean_reward", eval_mean_reward, t)
                self.logger.tb_writer_add_scalar("eval/mean_ep_length", eval_mean_ep_length, t)
                self.logger.tb_writer_add_scalar("eval/success_rate", eval_success_rate, t)
                self.logger.tb_writer_add_scalar("eval/state_change_rate", eval_state_change_rate, t)

                best_model_changed = False
                if self.model_save_measure == 'reward':
                    eval_measure = eval_mean_reward 
                else:
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
                                    "ratios [CL]: " + str([round(self.CL.c, 2)])])

                    if best_model_changed: message += " *" 
                    tqdm.write("[info] " + message)     

                # ROLLOUT
                self.logger.tb_writer_add_scalar("rollout/ep_rew_mean", safe_dq_mean(self.ep_rew_dq), t)
                self.logger.tb_writer_add_scalar("rollout/ep_len_mean", safe_dq_mean(self.ep_len_dq), t)
                self.logger.tb_writer_add_scalar("rollout/success_rate", safe_dq_mean(self.ep_success_dq), t)

                # STATE CHANGED
                self.logger.tb_writer_add_scalar("rollout/state_changed", safe_dq_mean(self.ep_state_changed_dq), t)

                # TRAIN
                self.logger.tb_writer_add_scalar('train/critic_loss', safe_dq_mean(self.loss_q_dq), t)
                self.logger.tb_writer_add_scalar("train/actor_loss", safe_dq_mean(self.loss_pi_dq), t)

                # HER
                self.logger.tb_writer_add_scalar("her/virtual_experience_added", safe_dq_mean(self.virtual_experience_dq), t)
                
                # PER
                if self.per_active:
                    self.logger.tb_writer_add_scalar("per/beta", np.mean(self.replay_buffer.get_beta()), t)

                # CL
                self.logger.tb_writer_add_scalar("cl/c", self.CL.c, t)
             
                # HiER
                if self.HiER.lambda_mode != 'multifix':
                    self.logger.tb_writer_add_scalar("hier/buffer_size", self.HiER.replay_buffer.size, t)
                    self.logger.tb_writer_add_scalar("hier/lambda", self.HiER.lambda_t, t)
                    self.logger.tb_writer_add_scalar("hier/xi", self.HiER.xi, t)
                    self.logger.tb_writer_add_scalar("hier/batch_size", self.HiER.hier_batch_size, t)
                else:
                    for bin_index in range(self.HiER.bin_num): 
                        self.logger.tb_writer_add_scalar("hier/buffer_size_"+str(bin_index), self.HiER.replay_buffers[bin_index].size, t)
                        self.logger.tb_writer_add_scalar("hier/lambda"+str(bin_index), self.HiER.lambda_ts[bin_index], t)
                        self.logger.tb_writer_add_scalar("hier/xi_"+str(bin_index), self.HiER.xis[bin_index], t)       
                        self.logger.tb_writer_add_scalar("hier/batch_size_"+str(bin_index), self.HiER.hier_batch_sizes[bin_index], t)      

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
   
    