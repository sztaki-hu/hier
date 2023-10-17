from rltrain.buffers.replay import ReplayBuffer

class HL:
    def __init__(self,config):

        self.hl_threshold = 0
        self.hl_active = True 

        self.batch_ratio_mode = config['buffer']['highlights']['batch_ratio_mode']
        assert self.batch_ratio_mode in ['fix', 'prioritized']
        self.batch_ratio_prioritized_alpha = config['buffer']['highlights']['batch_ratio_prioritized_alpha']

        self.batch_size = config['trainer']['batch_size']
        self.hl_batch_ratio = config['buffer']['highlights']['batch_ratio']
        self.hl_buffer_size = config['buffer']['highlights']['buffer_size']
        self.hl_replay_buffer = ReplayBuffer(
                obs_dim=int(config['environment']['obs_dim']), 
                act_dim=int(config['environment']['act_dim']), 
                size=int(float(config['buffer']['highlights']['buffer_size'])))
        self.hl_success_cond = config['buffer']['highlights']['success_cond']

        self.hl_batch_size =  int(self.batch_size * self.hl_batch_ratio)
        
        
            
    def store_episode(self,episode, info_success):
        pass

    def update_priority(self,batch_priorities,replay_batch_size):
        if self.batch_ratio_mode == 'fix':
            pass

        elif self.batch_ratio_mode == 'prioritized':
            prio_hier = sum(batch_priorities[:replay_batch_size])
            prio_er = sum(batch_priorities[replay_batch_size:])
            
            sum_priroty = prio_er**self.batch_ratio_prioritized_alpha + prio_hier**self.batch_ratio_prioritized_alpha 

            prob_hier = prio_hier**self.batch_ratio_prioritized_alpha / sum_priroty
            #prob_er = prio_er**self.batch_ratio_prioritized_alpha / sum_priroty

            self.hl_batch_ratio = prob_hier    
            self.hl_batch_size =  int(self.batch_size * self.hl_batch_ratio) 
 





