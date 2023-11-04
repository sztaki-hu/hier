import numpy as np
import torch
from typing import Dict
class PrioritizedReplay(object):
    """
    Proportional Prioritization
    """
    def __init__(self, size: int, alpha: float = 0.6, beta_start: float = 0.4, beta_frames: int = 100000) -> None:
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1 #for beta calculation
        self.size = size
        self.buffer     = []
        self.pos        = 0
        self.priorities = np.zeros((size,), dtype=np.float32)
    
    def beta_by_frame(self, frame_idx: int) -> float:
        """
        Linearly increases beta from beta_start to 1 over time from 1 to beta_frames.
        
        3.4 ANNEALING THE BIAS (Paper: PER)
        We therefore exploit the flexibility of annealing the amount of importance-sampling
        correction over time, by defining a schedule on the exponent 
        that reaches 1 only at the end of learning. In practice, we linearly anneal from its initial value 0 to 1
        """
        return min(1.0, self.beta_start + frame_idx * (1.0 - self.beta_start) / self.beta_frames)
    
    # def push(self, state, action, reward, next_state, done):
    def store(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, done: bool) -> None:
    
        assert state.ndim == next_state.ndim
        state      = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
        
        max_prio = self.priorities.max() if self.buffer else 1.0 # gives max priority if buffer is not empty else 1
        
        if len(self.buffer) < self.size:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            # puts the new data on the position of the oldes since it circles via pos variable
            # since if len(buffer) == size -> pos == 0 -> oldest memory (at least for the first round?) 
            self.buffer[self.pos] = (state, action, reward, next_state, done) 
        
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.size # lets the pos circle in the ranges of size if pos+1 > cap --> new posi = 0
    
    # def sample(self, batch_size):
    def sample_batch(self, batch_size: int) -> Dict:
        N = len(self.buffer)
        if N == self.size:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]
            
        # calc P = p^a/sum(p^a)
        probs  = prios ** self.alpha
        P = probs/probs.sum()
        
        #gets the indices depending on the probability p
        indices = np.random.choice(N, batch_size, p=P) 
        samples = [self.buffer[idx] for idx in indices]
        
        beta = self.beta_by_frame(self.frame)
        self.frame+=1
                
        #Compute importance-sampling weight
        weights  = (N * P[indices]) ** (-beta)
        # normalize weights
        weights /= weights.max() 
        weights  = np.array(weights, dtype=np.float32) 

        states, actions, rewards, next_states, dones = zip(*samples) 

        batch = dict(obs=np.concatenate(states),
                     obs2=np.concatenate(next_states),
                     act=actions,
                     rew=rewards,
                     done=dones,
                     indices = indices,
                     weights=weights
                     )
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}

    
    def update_priorities(self, batch_indices: np.ndarray, batch_priorities: np.ndarray) -> None:

        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = abs(prio) 

    def __len__(self) -> int:
        return len(self.buffer)

    def get_all(self) -> Dict:
        states, actions, rewards, next_states, dones = zip(*self.buffer) 
        batch = dict(obs=np.concatenate(states),
                     obs2=np.concatenate(next_states),
                     act=actions,
                     rew=rewards,
                     done=dones,
                     )
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}
    
    def get_beta(self) -> float:
        return self.beta_by_frame(self.frame)
    
