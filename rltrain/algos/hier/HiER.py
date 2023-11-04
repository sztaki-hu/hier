import numpy as np
from typing import Dict, List, Tuple
Transition = Tuple[np.ndarray,np.ndarray,float,np.ndarray,bool]

from rltrain.buffers.replay import ReplayBuffer

class HiER:
    def __init__(self,config: Dict) -> None:

        self.lambda_t = 0
        self.active = True 

        self.lambda_mode = config['buffer']['hier']['lambda']['mode']
        self.xi_mode = config['buffer']['hier']['xi']['mode']
        assert self.xi_mode in ['fix', 'prioritized']

        self.xi_prioritized_alpha = config['buffer']['hier']['xi']['prioritized']['alpha']

        self.batch_size = config['trainer']['batch_size']
        self.success_cond = config['buffer']['hier']['success_cond']

        if self.lambda_mode != 'multifix':   
            self.xi = config['buffer']['hier']['xi']['xi']
            self.buffer_size = config['buffer']['hier']['buffer_size']
            self.replay_buffer = ReplayBuffer(
                    obs_dim=int(config['environment']['obs_dim']), 
                    act_dim=int(config['environment']['act_dim']), 
                    size=int(float(config['buffer']['hier']['buffer_size'])))
            self.hier_batch_size =  int(self.batch_size * self.xi)
            self.xi_min = config['buffer']['hier']['xi']['prioritized']['xi_min']
            self.xi_max = config['buffer']['hier']['xi']['prioritized']['xi_max']
            
            self.bin_num = 0
            self.replay_buffers = []
            self.lambda_ts = []
            self.xis = []
            self.batch_sizes = []

            
    def store_episode(self, episode: List[Transition], info_success: bool, t: int) -> None:
        pass

    def update_priority(self, batch_priorities: np.ndarray, ser_batch_size: int) -> None:
        if self.xi_mode == 'fix':
            pass

        elif self.xi_mode == 'prioritized':
            batch_priorities = abs(batch_priorities)

            prio_hier = np.mean(batch_priorities[ser_batch_size:])
            prio_er = np.mean(batch_priorities[:ser_batch_size])

            sum_priroty = prio_er**self.xi_prioritized_alpha + prio_hier**self.xi_prioritized_alpha 

            prob_hier = prio_hier**self.xi_prioritized_alpha / sum_priroty
            #prob_er = prio_er**self.xi_prioritized_alpha / sum_priroty

            self.xi = min(max(prob_hier, self.xi_min),self.xi_max)

            self.hier_batch_size =  int(self.batch_size * self.xi) 

    
    def is_sampling_possible(self) -> bool:
        return True if self.replay_buffer.size > 0 else False
    
    def get_ser_batch_size(self) -> int:
        return int(self.batch_size - self.hier_batch_size)
    
    def sample_batch(self) -> Dict:
        return self.replay_buffer.sample_batch(self.hier_batch_size)
