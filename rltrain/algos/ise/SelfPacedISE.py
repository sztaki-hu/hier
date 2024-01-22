import numpy as np
import collections
from statistics import mean as dq_mean

from typing import Dict, List, Tuple, Union, Optional

from rltrain.taskenvs.GymPanda import GymPanda
from rltrain.algos.ise.InitialStateEntropy import InitialStateEntropy

class SelfPacedISE(InitialStateEntropy):

    def __init__(self, config: Dict, taskenv: GymPanda) -> None:
        super(SelfPacedISE, self).__init__(config, taskenv)

        # SELFPACED DUAL
        self.Psi_high = self.config['trainer']['ise']['selfpaced']['upper_cond']
        self.Psi_low = self.config['trainer']['ise']['selfpaced']['lower_cond']
        self.step = self.config['trainer']['ise']['selfpaced']['step']
        self.dequeu_maxlen = config['trainer']['ise']['selfpaced']['window_size']
      
        self.store_rollout_success_rate = True
        self.rollout_success_dq = collections.deque(maxlen=self.dequeu_maxlen)
        
   
    def update_c(self, t: int) -> None:
        if len(self.rollout_success_dq) == self.dequeu_maxlen: 
            success_rate = dq_mean(self.rollout_success_dq)
            if success_rate > self.Psi_high: 
                self.c += self.step
                self.c = min(self.c,1.0)
                self.rollout_success_dq.clear()
            elif success_rate < self.Psi_low:
                self.c -= self.step
                self.c = max(self.c,0.0)
                self.rollout_success_dq.clear()
                
            self.c_discard = max(0.0, self.c - self.c_discard_lag)
            self.copy_c_to_obj_and_goal()
    
         
    




     