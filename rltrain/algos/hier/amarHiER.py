import numpy as np
import collections
from statistics import mean as dq_mean
from typing import Dict, List, Tuple
Transition = Tuple[np.ndarray,np.ndarray,float,np.ndarray,bool]

from rltrain.algos.hier.HiER import HiER

class amarHiER(HiER): # Adaptive Moving Average Relative
    def __init__(self, config: Dict) -> None:
        super(amarHiER, self).__init__(config)

        self.lambda_start = config['buffer']['hier']['lambda']['amar']['lambda_start']
        self.lambda_end = config['buffer']['hier']['lambda']['amar']['lambda_end']
        self.lambda_margin_relative = config['buffer']['hier']['lambda']['amar']['lambda_margin_relative']
        assert self.lambda_margin_relative > 0
        self.window = config['buffer']['hier']['lambda']['amar']['window']

        self.lambda_t = self.lambda_start
        self.ep_rew_dq = collections.deque(maxlen=self.window)
    
    def store_episode(self, episode: List[Transition], info_success: bool, t: int) -> None:

        if self.success_cond and info_success == False: return

        if len(episode) <= 1: return
        
        sum_rew = 0
        for (o, a, r, o2, d) in episode:
            sum_rew += r

        if sum_rew >= self.lambda_t:
            for (o, a, r, o2, d) in episode:
                self.replay_buffer.store(o, a, r, o2, d)

            self.ep_rew_dq.append(sum_rew)
            if len(self.ep_rew_dq) == self.window: 
                margin = abs(dq_mean(self.ep_rew_dq) * self.lambda_margin_relative)
                self.lambda_t = min(self.lambda_end, dq_mean(self.ep_rew_dq) + margin)
