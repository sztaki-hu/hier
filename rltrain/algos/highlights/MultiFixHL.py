import numpy as np
import collections
import torch

from rltrain.algos.highlights.HL import HL
from rltrain.buffers.replay import ReplayBuffer

class MultiFixHL(HL):
    def __init__(self, config):
        super(MultiFixHL, self).__init__(config)

        self.hl_thresholds = config['buffer']['highlights']['multifix']['thresholds']
        self.hl_bin_num = len(self.hl_thresholds)
        
        self.hl_batch_ratios = config['buffer']['highlights']['multifix']['batch_ratios']
        self.hl_batch_sizes = [int(x * self.batch_size) for x in self.hl_batch_ratios]
        self.hl_buffer_sizes = [config['buffer']['highlights']['buffer_size']] * self.hl_bin_num
        self.hl_batch_bin_min_sample = int(config['buffer']['highlights']['multifix']['batch_bin_min_sample'])

        self.hl_replay_buffers = []
        for i in range(self.hl_bin_num):
             self.hl_replay_buffers.append(ReplayBuffer(
                    obs_dim=int(config['environment']['obs_dim']), 
                    act_dim=int(config['environment']['act_dim']), 
                    size=int(float(config['buffer']['highlights']['buffer_size']))))
             
    
    def store_episode(self,episode,info_success,t):

        if self.hl_success_cond and info_success == False: return

        if self.is_sampling_possible() == False:
            for i in range(self.hl_bin_num):
                for (o, a, r, o2, d) in episode:
                    self.hl_replay_buffers[i].store(o, a, r, o2, d)   
            return      

        sum_rew = 0
        for (o, a, r, o2, d) in episode:
            sum_rew += r
        
        for i in range(self.hl_bin_num):
            if sum_rew >= self.hl_thresholds[i]:
                for (o, a, r, o2, d) in episode:
                    self.hl_replay_buffers[i].store(o, a, r, o2, d)
                break
    
    def update_priority(self,batch_priorities,replay_batch_size):
        if self.batch_ratio_mode == 'fix':
            pass

        elif self.batch_ratio_mode == 'prioritized':

            prios = []
            
            prios.append(np.mean(batch_priorities[replay_batch_size:]))

            start_i = replay_batch_size 
            for i in range(self.hl_bin_num):
                end_i = start_i + self.hl_batch_sizes[i]
                prios.append(np.mean(batch_priorities[start_i:end_i]))
                start_i = end_i


            sum_prios = 0
            for prio in prios:
                sum_prios += prio**self.batch_ratio_prioritized_alpha
            
            probs = []
            for prio in prios:
                probs.append(prio**self.batch_ratio_prioritized_alpha / sum_prios)

            self.hl_batch_ratios = probs[1:]

            self.hl_batch_sizes = [int(min(max(x * self.batch_size,self.hl_batch_bin_min_sample),self.batch_size-self.hl_batch_bin_min_sample)) for x in self.hl_batch_ratios]
    
    def is_sampling_possible(self):
        for i in range(self.hl_bin_num):
            if self.hl_replay_buffers[i].size > 0: return True
        return False
    
    def get_replay_batch_size(self):
        return int(self.batch_size - sum(self.hl_batch_sizes))

    def sample_batch(self):
        first_batch = True
        for i in range(self.hl_bin_num):
            if self.hl_batch_sizes[i] > 0 and self.hl_replay_buffers[i].size > 0: 
                if first_batch:
                    batch = self.hl_replay_buffers[i].sample_batch(self.hl_batch_sizes[i])
                    first_batch = False
                else:
                    pivot = self.hl_replay_buffers[i].sample_batch(self.hl_batch_sizes[i])
                    batch = dict(obs=torch.cat((batch['obs'], pivot['obs']), 0),
                                    obs2=torch.cat((batch['obs2'], pivot['obs2']), 0),
                                    act=torch.cat((batch['act'], pivot['act']), 0),
                                    rew=torch.cat((batch['rew'], pivot['rew']), 0),
                                    done=torch.cat((batch['done'], pivot['done']), 0),
                                    indices=torch.cat((batch['indices'], pivot['indices']), 0),
                                    weights=torch.cat((batch['weights'], pivot['weights']), 0))
        return batch
            

        
