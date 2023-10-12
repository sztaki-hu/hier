from rltrain.buffers.replay import ReplayBuffer
from rltrain.algos.highlights.HL import HL

class noHL(HL):
    def __init__(self, config):

        self.hl_batch_ratio = 0
        self.hl_buffer_size = 0
        self.hl_replay_buffer = ReplayBuffer(
                obs_dim=int(config['environment']['obs_dim']), 
                act_dim=int(config['environment']['act_dim']), 
                size=0)
        
        self.hl_threshold = 0
        self.hl_active = False 
    
    

