import numpy as np
import torch
import time
import collections
import random

from tqdm import tqdm

from rltrain.envs.builder import make_env

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

        self.print_out_name = '_'.join((self.config['general']['exp_name'],
                                        self.config['environment']['task']['name'],
                                        self.config['agent']['type'],
                                        self.her_goal_selection_strategy,
                                        str(main_args.trainid)))

        
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
    
    # def get_action(self, o, deterministic=False):
    #     return self.agent.ac.act(torch.as_tensor(o, dtype=torch.float32), deterministic)
    
    def get_new_goals(self, episode, ep_t):
        if self.her_goal_selection_strategy == 'final':
            new_goals = []
            _, _, _, o2, _ = episode[-1]
            for _ in range(self.her_n_sampled_goal):
                new_goals.append(self.env.get_goal_state_from_obs(o2))
            return new_goals
        elif self.her_goal_selection_strategy == 'future' or self.her_goal_selection_strategy == 'future_once':
            new_goals = []
            for _ in range(self.her_n_sampled_goal):
                rand_future_transition = random.randint(ep_t, len(episode)-1)
                _, _, _, o2, _ = episode[rand_future_transition]
                new_goals.append(self.env.get_goal_state_from_obs(o2))
            return new_goals
        elif self.her_goal_selection_strategy == 'near':
            new_goals = []
            for _ in range(self.her_n_sampled_goal):
                rand_future_transition = random.randint(ep_t, min(len(episode)-1,ep_t+5))
                _, _, _, o2, _ = episode[rand_future_transition]
                new_goals.append(self.env.get_goal_state_from_obs(o2))
            return new_goals
        elif self.her_goal_selection_strategy == 'next':
            new_goals = []
            for _ in range(self.her_n_sampled_goal):
                _, _, _, o2, _ = episode[ep_t]
                new_goals.append(self.env.get_goal_state_from_obs(o2))
            return new_goals

    def test_agent(self):
        ep_rets = []
        ep_lens = []
        success_num = 0.0
        for j in range(self.eval_num_episodes):
            [o, info], d, ep_ret, ep_len = self.env.reset_with_init_check(), False, 0, 0
            while not(d or (ep_len == self.max_ep_len)):
                # Take deterministic actions at test time 
                if self.agent_type == 'sac':
                    a = self.agent.get_action(o, True)
                elif self.agent_type == 'td3':
                    a = self.agent.get_action(o, 0)
                o, r, terminated, truncated, info = self.env.step(a)
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
        
        self.agent = agent

        init_invalid_num = 0
        reset_num = 0

        [o, reset_info], ep_ret, ep_len = self.env.reset_with_init_check(), 0, 0
        init_invalid_num += reset_info['init_invalid_num']
        reset_num += reset_info['reset_num']

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

                self.ep_success_dq.append(1.0) if info['is_success'] == True else self.ep_success_dq.append(0.0) 
                
                for (o, a, r, o2, d) in episode:
                    replay_buffer.store(o, a, r, o2, d)
                
                if self.her_active and truncated:
                    if self.her_goal_selection_strategy == 'future_once':
                        new_goals = self.get_new_goals(episode,0)
                        for (o, a, r, o2, d) in episode:                  
                            for new_goal in new_goals:
                                o_new = self.env.change_goal_in_obs(o, new_goal)
                                o2_new = self.env.change_goal_in_obs(o2, new_goal)
                                r_new, d_new = self.env.her_get_reward_and_done(o2_new) 
                                replay_buffer.store(o_new, a, r_new, o2_new, d_new)
                    else:
                        ep_t = 0
                        for (o, a, r, o2, d) in episode:
                            new_goals = self.get_new_goals(episode,ep_t)
                            for new_goal in new_goals:
                                o_new = self.env.change_goal_in_obs(o, new_goal)
                                o2_new = self.env.change_goal_in_obs(o2, new_goal)
                                r_new, d_new = self.env.her_get_reward_and_done(o2_new) 
                                replay_buffer.store(o_new, a, r_new, o2_new, d_new)
                            ep_t += 1
                episode = []

                # print("-------------------")
                # print(replay_buffer.get_all())
                # assert False

                # logger.store(EpRet=ep_ret, EpLen=ep_len)
                # self.logger.tb_writer_add_scalar("train/ep_ret", ep_ret, t)
                # self.logger.tb_writer_add_scalar("train/ep_len", ep_len, t)
                self.ep_rew_dq.append(ep_ret)
                self.ep_len_dq.append(ep_len)
                [o, reset_info], ep_ret, ep_len = self.env.reset_with_init_check(), 0, 0
                init_invalid_num += reset_info['init_invalid_num']
                reset_num += reset_info['reset_num']

            # Update handling
            if t >= self.update_after and t % self.update_every == 0:
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
                if (epoch % self.model_save_freq == 0) or best_model_changed:
                    model_path = self.logger.get_model_save_path(epoch)
                    self.agent.save_model(model_path,self.model_save_mode)
                    message = self.print_out_name +  " | t: " + str(t) +  " | epoch: " + str(epoch) + " | eval_mean_reward: " + str(eval_mean_reward) + " | eval_mean_ep_length: " + str(eval_mean_ep_length) + " | eval_success_rate: " + str(eval_success_rate)
                    if best_model_changed: message += " *"
                    tqdm.write("[info] " + message)
                    #logger.save_state({'env': env}, None)       

                # ROLLOUT
                self.logger.tb_writer_add_scalar("rollout/ep_rew_mean", np.mean(self.ep_rew_dq), t)
                self.logger.tb_writer_add_scalar("rollout/ep_len_mean", np.mean(self.ep_len_dq), t)
                self.logger.tb_writer_add_scalar("rollout/success_rate", np.mean(self.ep_success_dq), t)

                # TRAIN
                self.logger.tb_writer_add_scalar('train/critic_loss', np.mean(self.loss_q_dq), t)
                self.logger.tb_writer_add_scalar("train/actor_loss", np.mean(self.loss_pi_dq), t)

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
   
    