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
    
    def get_t(self):
        return self.t

    def get_ptr(self):
        return self.ptr