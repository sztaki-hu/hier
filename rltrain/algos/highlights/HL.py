from rltrain.buffers.replay import ReplayBuffer

class HL:
    def __init__(self,config):

        self.hl_threshold = 0
        self.hl_active = True 
        
        self.hl_batch_ratio = config['buffer']['highlights']['batch_ratio']
        self.hl_buffer_size = config['buffer']['highlights']['buffer_size']
        self.hl_replay_buffer = ReplayBuffer(
                obs_dim=int(config['environment']['obs_dim']), 
                act_dim=int(config['environment']['act_dim']), 
                size=int(float(config['buffer']['highlights']['buffer_size'])))
        self.hl_success_cond = config['buffer']['highlights']['success_cond']
            
    def store_episode(self,episode, info_success):
        pass
