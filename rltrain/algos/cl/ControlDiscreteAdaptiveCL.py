import numpy as np
import collections
import math

from rltrain.algos.cl.CL import CL

class ControlDiscreteAdaptiveCL(CL):

    def __init__(self, config, env, replay_buffer):
        super(ControlDiscreteAdaptiveCL, self).__init__(config, env, replay_buffer)

        # Control Discrete Adaptive
        self.cl_target_plus_eval = self.config['trainer']['cl']['controldiscreteadaptive']['target_plus_eval']
        self.cl_step = self.config['trainer']['cl']['controldiscreteadaptive']['step']
        self.cl_eval_dq_maxlen = self.config['trainer']['cl']['controldiscreteadaptive']['window_size_eval']
        self.cl_rollout_dq_maxlen = self.config['trainer']['cl']['controldiscreteadaptive']['window_size_rollout']
       
        self.store_rollout_success_rate = True
        self.store_eval_success_rate = True
        self.cl_eval_success_dq = collections.deque(maxlen=self.cl_eval_dq_maxlen)
        self.cl_rollout_success_dq = collections.deque(maxlen=self.cl_rollout_dq_maxlen) 
   
    def update_cl(self,t):  
        success_rate = np.mean(self.cl_rollout_success_dq) if len(self.cl_rollout_success_dq) > 0  else 0
        eval_success_rate = np.mean(self.cl_eval_success_dq) if len(self.cl_eval_success_dq) > 0 else 0
        target = eval_success_rate + self.cl_target_plus_eval
        if target > success_rate: 
            self.cl_ratio -= self.cl_step
            self.cl_ratio = max(self.cl_ratio,0.0)
            self.copy_cl_ratios_to_obj_and_goal()
        else:
            self.cl_ratio += self.cl_step
            self.cl_ratio = min(self.cl_ratio,1.0)
            self.copy_cl_ratios_to_obj_and_goal()
    


           
    




     