import numpy as np
import collections
import math
from statistics import mean as dq_mean

from typing import Dict, List, Tuple, Union, Optional

from rltrain.taskenvs.GymPanda import GymPanda
from rltrain.algos.initstate.ise.InitStateEntropy import InitStateEntropy

class ControlISE(InitStateEntropy):

    def __init__(self, config: Dict, taskenv: GymPanda) -> None:
        super(ControlISE, self).__init__(config, taskenv)

        # Control
        self.target_value = self.config['trainer']['init_state']['ise']['control']['target_value']
        self.step = self.config['trainer']['init_state']['ise']['control']['step']
        self.dequeu_maxlen = config['trainer']['init_state']['ise']['control']['window_size']
      
        self.store_rollout_success_rate = True
        self.rollout_success_dq = collections.deque(maxlen=self.dequeu_maxlen)
   
    def update_c(self, t: int) -> None:  
        success_rate = dq_mean(self.rollout_success_dq) if t > 0 else 0
        if self.target_value > success_rate: 
            self.c -= self.step
            self.c = max(self.c,0.0)
            self.copy_c_to_obj_and_goal()
        else:
            self.c += self.step
            self.c = min(self.c,1.0)
            self.copy_c_to_obj_and_goal()
        
        self.c_discard = max(0.0, self.c - self.c_discard_lag)
        


           
    




     