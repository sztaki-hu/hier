from rltrain.algos.hier.HiER import HiER
import numpy as np
from typing import Dict, List, Tuple
Transition = Tuple[np.ndarray,np.ndarray,float,np.ndarray,bool]

class fixHiER(HiER):
    def __init__(self, config: Dict) -> None:
        super(fixHiER, self).__init__(config)

        self.lambda_t = config['buffer']['hier']['lambda']['fix']['lambda']
    
    
    def store_episode(self, episode: List[Transition], info_success: bool, t: int) -> None:

        if self.success_cond and info_success == False: return

        sum_rew = 0
        for (o, a, r, o2, d) in episode:
            sum_rew += r

        if sum_rew >= self.lambda_t:
            for (o, a, r, o2, d) in episode:
                self.replay_buffer.store(o, a, r, o2, d)
