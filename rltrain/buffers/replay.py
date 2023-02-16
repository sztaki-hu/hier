import numpy as np
import torch

from rltrain.agents.core import combined_shape

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)

        self.rew_nstep_buff = np.zeros(size, dtype=np.float32)
        self.obs_nstep_buff = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.done_nstep_buf = np.zeros(size, dtype=np.float32)
        self.n_nstep_buf = np.zeros(size, dtype=np.float32)

        self.ptr, self.size, self.max_size = 0, 0, size
        self.t = 0

    def store(self, obs, act, rew, next_obs, done, r_nstep = None, obs_nstep = None, done_nstep = None, n_nstep = None):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done

        self.rew_nstep_buff[self.ptr] = r_nstep
        self.obs_nstep_buff[self.ptr] = obs_nstep
        self.done_nstep_buf[self.ptr] = done_nstep
        self.n_nstep_buf[self.ptr] = n_nstep

        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)
        self.t += 1
    
    def store_episode_nstep(self,ep_transitions,n_step,gamma):
        for i in range(len(ep_transitions)):
            o, a, r, o2, d = ep_transitions[i]
            r_nstep = ep_transitions[i][2]
            obs_nstep = ep_transitions[i][3]
            d_nstep = ep_transitions[i][4]
            j = 0
            if d_nstep == 0:
                for j in range(1,n_step):
                    if i + j < len(ep_transitions):
                        r_nstep += ep_transitions[i+j][2] * gamma**j
                        obs_nstep = ep_transitions[i+j][3]
                        d_nstep = ep_transitions[i+j][4]
                        if d_nstep == 1:
                            break
            n_nstep = j
            self.store(o, a, r, o2, d, r_nstep, obs_nstep, d_nstep, n_nstep)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs],
                     rew_nstep=self.rew_nstep_buff[idxs],
                     obs_nstep=self.obs_nstep_buff[idxs],
                     done_nstep=self.done_nstep_buf[idxs],
                     n_nstep=self.n_nstep_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}
    
    def get_all(self):
        idxs = np.arange(self.size, dtype=int)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs],
                     rew_nstep=self.rew_nstep_buff[idxs],
                     obs_nstep=self.obs_nstep_buff[idxs],
                     done_nstep=self.done_nstep_buf[idxs],
                     n_nstep=self.n_nstep_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}
    
    def get_first(self, num):
        max_sample = min(int(num),self.size)
        idxs = np.arange(max_sample, dtype=int)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs],
                     rew_nstep=self.rew_nstep_buff[idxs],
                     obs_nstep=self.obs_nstep_buff[idxs],
                     done_nstep=self.done_nstep_buf[idxs],
                     n_nstep=self.n_nstep_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}
    
    def get_t(self):
        return self.t

    def get_ptr(self):
        return self.ptr