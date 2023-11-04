from typing import Dict

from rltrain.buffers.replay import ReplayBuffer
from rltrain.algos.hier.HiER import HiER

class noHiER(HiER):
    def __init__(self, config: Dict) -> None:

        self.xi = 0
        self.batch_size =  0
        self.buffer_size = 0
        self.replay_buffer = ReplayBuffer(
                obs_dim=int(config['environment']['obs_dim']), 
                act_dim=int(config['environment']['act_dim']), 
                size=0)
        
        self.lambda_t = 0
        self.active = False 
        self.mode = config['buffer']['highlights']['mode']
    
    

