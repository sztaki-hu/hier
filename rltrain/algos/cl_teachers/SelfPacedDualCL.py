import numpy as np
import collections

from rltrain.algos.cl_teachers.CL import CL

class SelfPacedDualCL(CL):

    def __init__(self, config, env, replay_buffer):
        super(SelfPacedDualCL, self).__init__(config, env, replay_buffer)

        # SELFPACED DUAL
        self.cl_upper_cond = self.config['trainer']['cl']['selfpaceddual']['upper_cond']
        self.cl_lower_cond = self.config['trainer']['cl']['selfpaceddual']['lower_cond']
        self.cl_step = self.config['trainer']['cl']['selfpaceddual']['step']
        self.cl_dequeu_maxlen = config['trainer']['cl']['selfpaceddual']['window_size']
        self.cl_ratio = 0 
        self.cl_ratio_discard = 0
        self.cl_ep_success_dq = collections.deque(maxlen=self.cl_dequeu_maxlen)
        self.store_success_rate = True
   
    def update_cl(self,t):
        if len(self.cl_ep_success_dq) == self.cl_dequeu_maxlen: 
            success_rate = np.mean(self.cl_ep_success_dq)
            if success_rate > self.cl_upper_cond: 
                self.cl_ratio += self.cl_step
                self.cl_ratio = min(self.cl_ratio,1.0)
                self.cl_ep_success_dq.clear()
            elif success_rate < self.cl_lower_cond:
                self.cl_ratio -= self.cl_step
                self.cl_ratio = max(self.cl_ratio,0.0)
                self.cl_ep_success_dq.clear()
            self.cl_ratio_discard = max(0.0, self.cl_ratio - self.cl_ratio_discard_lag)
    
         
    




     