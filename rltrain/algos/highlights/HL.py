import numpy as np

from rltrain.buffers.replay import ReplayBuffer

class HL:
    def __init__(self,config):

        self.hl_threshold = 0
        self.hl_active = True 
        self.hl_mode = config['buffer']['highlights']['mode']

        self.batch_ratio_mode = config['buffer']['highlights']['batch_ratio_mode']
        assert self.batch_ratio_mode in ['fix', 'prioritized']
        self.batch_ratio_prioritized_alpha = config['buffer']['highlights']['batch_ratio_prioritized_alpha']

        self.batch_size = config['trainer']['batch_size']
        self.hl_success_cond = config['buffer']['highlights']['success_cond']

        if self.hl_mode != 'multifix':   
            self.hl_batch_ratio = config['buffer']['highlights']['batch_ratio']
            self.hl_buffer_size = config['buffer']['highlights']['buffer_size']
            self.hl_replay_buffer = ReplayBuffer(
                    obs_dim=int(config['environment']['obs_dim']), 
                    act_dim=int(config['environment']['act_dim']), 
                    size=int(float(config['buffer']['highlights']['buffer_size'])))
            self.hl_batch_size =  int(self.batch_size * self.hl_batch_ratio)
            self.hl_batch_ratio_min = config['buffer']['highlights']['batch_ratio_min']
            self.hl_batch_ratio_max = config['buffer']['highlights']['batch_ratio_max']
        
            
    def store_episode(self,episode, info_success,t):
        pass

    def update_priority(self,batch_priorities,replay_batch_size):
        if self.batch_ratio_mode == 'fix':
            pass

        elif self.batch_ratio_mode == 'prioritized':
            batch_priorities = abs(batch_priorities)

            prio_hier = np.mean(batch_priorities[replay_batch_size:])
            prio_er = np.mean(batch_priorities[:replay_batch_size])

            sum_priroty = prio_er**self.batch_ratio_prioritized_alpha + prio_hier**self.batch_ratio_prioritized_alpha 

            prob_hier = prio_hier**self.batch_ratio_prioritized_alpha / sum_priroty
            #prob_er = prio_er**self.batch_ratio_prioritized_alpha / sum_priroty

            self.hl_batch_ratio = min(max(prob_hier, self.hl_batch_ratio_min),self.hl_batch_ratio_max)

            self.hl_batch_size =  int(self.batch_size * self.hl_batch_ratio) 

    
    def is_sampling_possible(self):
        if self.hl_replay_buffer.size > 0: return True
    
    def get_replay_batch_size(self):
        return int(self.batch_size - self.hl_batch_size)
    
    def sample_batch(self):
        return self.hl_replay_buffer.sample_batch(self.hl_batch_size)
