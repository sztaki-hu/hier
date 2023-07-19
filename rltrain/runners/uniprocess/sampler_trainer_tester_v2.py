import numpy as np
import torch
import time
import collections

from tqdm import tqdm

from rltrain.envs.builder import make_env

class SamplerTrainerTester:

    def __init__(self,device,logger,config):

        self.device = device
        
        self.logger = logger
        self.config = config

        self.seed = config['general']['seed'] 

        self.steps_per_epoch = config['trainer']['steps_per_epoch'] 
        self.epochs = config['trainer']['epochs'] 

        self.replay_size = int(config['buffer']['replay_buffer_size']) 
        self.batch_size = config['trainer']['batch_size'] 
        
        self.update_after = config['trainer']['update_after'] 
        self.update_every = config['trainer']['update_every'] 

        self.obs_dim = config['environment']['obs_dim']
        self.act_dim = config['environment']['act_dim']
        self.boundary_min = np.array(config['agent']['boundary_min'])[:self.act_dim]
        self.boundary_max = np.array(config['agent']['boundary_max'])[:self.act_dim]

        self.demo_use = config['demo']['demo_use']  

        self.start_steps = config['sampler']['start_steps']
        self.max_ep_len = config['sampler']['max_ep_len']
        self.num_test_episodes = config['tester']['num_test_episodes']

        self.model_save_freq = config['logger']['model']['save']['freq']
        self.model_save_best_after = config['logger']['model']['save']['best_after']
        self.model_save_mode = config['logger']['model']['save']['mode']

        self.train_ep_ret_dq = collections.deque(maxlen=10)
        self.train_ep_len_dq = collections.deque(maxlen=10)
        self.loss_q_dq = collections.deque(maxlen=10)
        self.loss_pi_dq = collections.deque(maxlen=10)

        
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

    def test_agent(self):
        for j in range(self.num_test_episodes):
            o, d, ep_ret, ep_len = self.test_env.reset(), False, 0, 0
            while not(d or (ep_len == self.max_ep_len)):
                # Take deterministic actions at test time 
                o, r, d, _ = self.test_env.step(self.agent.get_action(o, True))
                ep_ret += r
                ep_len += 1
            #logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)
        
        return ep_ret, ep_len
            

    def start(self,agent,replay_buffer):

        self.env = make_env(self.config)
        self.test_env = make_env(self.config)
        self.agent = agent

        # Prepare for interaction with environment
        total_steps = self.steps_per_epoch * self.epochs
        #start_time = time.time()
        o, ep_ret, ep_len = self.env.reset(), 0, 0

        best_test_ep_ret = -float('inf')

        # Main loop: collect experience in env and update/log each epoch
        for t in tqdm(range(total_steps), desc ="Training: ", leave=True):
            
            # Until start_steps have elapsed, randomly sample actions
            # from a uniform distribution for better exploration. Afterwards, 
            # use the learned policy. 
            if t > self.start_steps:
                a = self.agent.get_action(o, False) 
                # a = self.get_action(o)
            else:
                a = self.env.random_sample()
                # a = self.env.env.action_space.sample()

            # Step the env
            o2, r, d, _ = self.env.step(a)
            ep_ret += r
            ep_len += 1

            # Ignore the "done" signal if it comes from hitting the time
            # horizon (that is, when it's an artificial terminal signal
            # that isn't based on the agent's state)
            d = False if ep_len==self.max_ep_len else d

            # Store experience to replay buffer
            replay_buffer.store(o, a, r, o2, d)

            # Super critical, easy to overlook step: make sure to update 
            # most recent observation!
            o = o2

            # End of trajectory handling
            if d or (ep_len == self.max_ep_len):
                # logger.store(EpRet=ep_ret, EpLen=ep_len)
                # self.logger.tb_writer_add_scalar("train/ep_ret", ep_ret, t)
                # self.logger.tb_writer_add_scalar("train/ep_len", ep_len, t)
                self.train_ep_ret_dq.append(ep_ret)
                self.train_ep_len_dq.append(ep_len)
                o, ep_ret, ep_len = self.env.reset(), 0, 0

            # Update handling
            if t >= self.update_after and t % self.update_every == 0:
                for j in range(self.update_every):
                    batch = replay_buffer.sample_batch(self.batch_size)
                    #print(batch)
                    ret_loss_q, ret_loss_pi = self.agent.update(data=batch)
                    self.loss_q_dq.append(ret_loss_q)
                    self.loss_pi_dq.append(ret_loss_pi)

            # End of epoch handling
            if (t+1) % self.steps_per_epoch == 0:
                epoch = (t+1) // self.steps_per_epoch

                # Test the performance of the deterministic version of the agent.
                test_ep_ret, test_ep_len = self.test_agent()

                self.logger.tb_writer_add_scalar("test/ep_ret", test_ep_ret, epoch)
                self.logger.tb_writer_add_scalar("test/ep_len", test_ep_len, epoch)

                best_model_changed = False
                if epoch > self.model_save_best_after:
                    if test_ep_ret > best_test_ep_ret:
                        best_test_ep_ret = test_ep_ret
                        best_model_changed = True
                
                self.logger.tb_writer_add_scalar("test/best_ep_len", best_test_ep_ret, epoch)

                # Save model 
                if (epoch % self.model_save_freq == 0) or (epoch == self.epochs) or best_model_changed:
                    model_path = self.logger.get_model_save_path(epoch)
                    self.agent.save_model(model_path,self.model_save_mode)
                    #logger.save_state({'env': env}, None)       

                self.logger.tb_writer_add_scalar("train/train_ep_ret_", np.mean(self.train_ep_ret_dq), t)
                self.logger.tb_writer_add_scalar("train/train_ep_len_", np.mean(self.train_ep_len_dq), t)
                self.logger.tb_writer_add_scalar('train/loss_q_', np.mean(self.loss_q_dq), t)
                self.logger.tb_writer_add_scalar("train/loss_pi_", np.mean(self.loss_pi_dq), t)
   
    