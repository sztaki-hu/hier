import numpy as np
import collections
import torch

from typing import Dict, List, Tuple
Transition = Tuple[np.ndarray,np.ndarray,float,np.ndarray,bool]

from rltrain.algos.hier.HiER import HiER
from rltrain.buffers.replay import ReplayBuffer

class multifixHiER(HiER):
    def __init__(self, config: Dict) -> None:
        super(multifixHiER, self).__init__(config)

        self.lambda_ts = config['buffer']['hier']['lambda']['multifix']['lambdas']
        self.bin_num = len(self.lambda_ts)
        
        self.xis = config['buffer']['hier']['lambda']['multifix']['xis']
        self.hier_batch_sizes = [int(x * self.batch_size) for x in self.xis]
        self.buffer_sizes = [config['buffer']['hier']['buffer_size']] * self.bin_num
        self.batch_bin_min_sample = int(config['buffer']['hier']['lambda']['multifix']['batch_bin_min_sample'])

        self.replay_buffers = []
        for i in range(self.bin_num):
             self.replay_buffers.append(ReplayBuffer(
                    obs_dim=int(config['environment']['obs_dim']), 
                    act_dim=int(config['environment']['act_dim']), 
                    size=int(float(config['buffer']['hier']['buffer_size']))))
             
    
    def store_episode(self, episode: List[Transition], info_success: bool, t: int) -> None:

        if self.success_cond and info_success == False: return

        if self.is_sampling_possible() == False:
            for i in range(self.bin_num):
                for (o, a, r, o2, d) in episode:
                    self.replay_buffers[i].store(o, a, r, o2, d)   
            return      

        sum_rew = 0
        for (o, a, r, o2, d) in episode:
            sum_rew += r
        
        for i in range(self.bin_num):
            if sum_rew >= self.lambda_ts[i]:
                for (o, a, r, o2, d) in episode:
                    self.replay_buffers[i].store(o, a, r, o2, d)
                break
    
    def update_priority(self, batch_priorities: np.ndarray, ser_batch_size: int) -> None:
        if self.xi_mode == 'fix':
            pass

        elif self.xi_mode == 'prioritized':

            batch_priorities = abs(batch_priorities)

            prios = []
            
            prios.append(np.mean(batch_priorities[:ser_batch_size]))

            start_i = ser_batch_size 
            for i in range(self.bin_num):
                end_i = start_i + self.hier_batch_sizes[i]
                prios.append(np.mean(batch_priorities[start_i:end_i]))
                start_i = end_i


            sum_prios = 0
            for prio in prios:
                sum_prios += prio**self.xi_prioritized_alpha
            
            probs = []
            for prio in prios:
                probs.append(prio**self.xi_prioritized_alpha / sum_prios)

            self.xis = probs[1:]

            self.hier_batch_sizes = [int(min(max(x * self.batch_size,self.batch_bin_min_sample),self.batch_size-self.batch_bin_min_sample)) for x in self.xis]
    
    def is_sampling_possible(self) -> bool:
        for i in range(self.bin_num):
            if self.replay_buffers[i].size > 0: return True
        return False
    
    def get_ser_batch_size(self) -> int:
        return int(self.batch_size - sum(self.hier_batch_sizes))

    def sample_batch(self) -> Dict:
        batch = {}
        first_batch = True
        for i in range(self.bin_num):
            if self.hier_batch_sizes[i] > 0 and self.replay_buffers[i].size > 0: 
                if first_batch:
                    batch = self.replay_buffers[i].sample_batch(self.hier_batch_sizes[i])
                    first_batch = False
                else:
                    pivot = self.replay_buffers[i].sample_batch(self.hier_batch_sizes[i])
                    batch = dict(obs=torch.cat((batch['obs'], pivot['obs']), 0),
                                    obs2=torch.cat((batch['obs2'], pivot['obs2']), 0),
                                    act=torch.cat((batch['act'], pivot['act']), 0),
                                    rew=torch.cat((batch['rew'], pivot['rew']), 0),
                                    done=torch.cat((batch['done'], pivot['done']), 0),
                                    indices=torch.cat((batch['indices'], pivot['indices']), 0),
                                    weights=torch.cat((batch['weights'], pivot['weights']), 0))
        return batch
            

        
