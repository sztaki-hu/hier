import math
import numpy as np
from typing import Dict, List, Tuple
Transition = Tuple[np.ndarray,np.ndarray,float,np.ndarray,bool]

from rltrain.algos.hier.HiER import HiER

PROFILES = ['linear','sqrt','quad']

class predefinedHiER(HiER):
    def __init__(self, config: Dict) -> None:
        super(predefinedHiER, self).__init__(config)

        self.lambda_profile = config['buffer']['hier']['lambda']['predefined']['lambda_profile']
        assert self.lambda_profile in PROFILES

        self.lambda_saturation_t = config['buffer']['hier']['lambda']['predefined']['lambda_saturation_t']
        assert 0.0 <= self.lambda_saturation_t <= 1.0
        
        self.lambda_start = config['buffer']['hier']['lambda']['predefined']['lambda_start']
        self.lambda_end = config['buffer']['hier']['lambda']['predefined']['lambda_end']
        self.lambda_t = self.lambda_start
        self.lambda_scalor = self.lambda_end - self.lambda_start
        assert self.lambda_scalor > 0 

        self.total_timesteps = float(config['trainer']['total_timesteps'])

        
    
    def store_episode(self,episode: List[Transition], info_success: bool, t: int) -> None:

        if self.success_cond and info_success == False: return

        self.update_lambda(t)

        sum_rew = 0
        for (o, a, r, o2, d) in episode:
            sum_rew += r

        if sum_rew >= self.lambda_t:
            for (o, a, r, o2, d) in episode:
                self.replay_buffer.store(o, a, r, o2, d)
    

    def update_lambda(self,t: int) -> None:
        t_rel = float(t) / self.total_timesteps
        lambda_rel01 = 0
        if self.lambda_profile == "linear":
            lambda_rel01 = min(1.0, t_rel / self.lambda_saturation_t)
        elif self.lambda_profile == "sqrt":
            lambda_rel01 = min(1.0, math.sqrt(t_rel / self.lambda_saturation_t))
        elif self.lambda_profile == "quad":
            lambda_rel01 = min(1.0, math.pow(t_rel / self.lambda_saturation_t,2))

        self.lambda_t = self.lambda_start + (self.lambda_scalor * lambda_rel01)

      

    
    
