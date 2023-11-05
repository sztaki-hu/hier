import numpy as np
import collections
import math
from statistics import mean as dq_mean

from typing import Dict, List, Tuple, Union, Optional

from rltrain.taskenvs.GymPanda import GymPanda
from rltrain.algos.cl.CL import CL

class controladaptiveCL(CL):

    def __init__(self, config: Dict, taskenv: GymPanda) -> None:
        super(controladaptiveCL, self).__init__(config, taskenv)

        # Control Discrete Adaptive
        self.Delta = self.config['trainer']['cl']['controladaptive']['Delta']
        self.target_max = self.config['trainer']['cl']['controladaptive']['target_max']
        self.step = self.config['trainer']['cl']['controladaptive']['step']
        self.eval_dq_maxlen = self.config['trainer']['cl']['controladaptive']['window_size_eval']
        self.rollout_dq_maxlen = self.config['trainer']['cl']['controladaptive']['window_size_rollout']
       
        self.store_rollout_success_rate = True
        self.store_eval_success_rate = True
        self.eval_success_dq = collections.deque(maxlen=self.eval_dq_maxlen)
        self.rollout_success_dq = collections.deque(maxlen=self.rollout_dq_maxlen) 
   
    def update_c(self,t: int) -> None: 
        success_rate = dq_mean(self.rollout_success_dq) if len(self.rollout_success_dq) > 0  else 0
        eval_success_rate = dq_mean(self.eval_success_dq) if len(self.eval_success_dq) > 0 else 0
        target = min(self.target_max, eval_success_rate + self.Delta)
        if target > success_rate: 
            self.c -= self.step
            self.c = max(self.c,0.0)
            self.copy_c_to_obj_and_goal()
        else:
            self.c += self.step
            self.c = min(self.c,1.0)
            self.copy_c_to_obj_and_goal()
    


           
    




     