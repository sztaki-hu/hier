from rltrain.buffers.replay import ReplayBuffer

HIGHLIGHTS_MODES = ['fix_th']

class Highlights:
    def __init__(self,config):
        self.highlights_active = config['buffer']['highlights']['bool']
        self.highlights_batch_ratio = config['buffer']['highlights']['batch_ratio']
        self.highlights_buffer_size = config['buffer']['highlights']['buffer_size']
        self.highlights_mode = config['buffer']['highlights']['mode']

        assert self.highlights_mode in HIGHLIGHTS_MODES

        if self.highlights_mode == 'fix_th': self.highlights_threshold = config['buffer']['highlights']['fix_th']['threshold']
    
        self.highlights_replay_buffer = ReplayBuffer(
                obs_dim=int(config['environment']['obs_dim']), 
                act_dim=int(config['environment']['act_dim']), 
                size=int(float(config['buffer']['highlights']['buffer_size'])))
    
    def store_episode(self,episode):
        if self.highlights_active:
            if self.highlights_mode == 'fix_th':
                if len(episode) <= self.highlights_threshold:
                    for (o, a, r, o2, d) in episode:
                        self.highlights_replay_buffer.store(o, a, r, o2, d)
