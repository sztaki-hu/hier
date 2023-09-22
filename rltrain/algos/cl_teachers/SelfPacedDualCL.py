import numpy as np
import collections

from rltrain.algos.cl_teachers.CL import CL

class SelfPacedDualCL(CL):

    def __init__(self, config, env, replay_buffer):
        super(SelfPacedDualCL, self).__init__(config, env, replay_buffer)

        # SELFPACED DUAL
        self.cl_upper_cond = self.config['trainer']['cl']['selfpaceddual']['upper_cond']
        self.cl_lower_cond = self.config['trainer']['cl']['selfpaceddual']['lower_cond']
        self.cl_step = self.config['trainer']['cl']['selfpaced']['step']
        self.cl_ratio = 0 
        self.cl_ep_success_dq = collections.deque(maxlen=config['trainer']['cl']['selfpaceddual']['window_size'])
        self.store_success_rate = True

        self.clear_cl_ep_success_dq()
   
    def update_ratio(self,t):
        success_rate = np.mean(self.cl_ep_success_dq) if t > 0 else 0

        if success_rate > self.cl_upper_cond: 
            self.cl_ratio += self.cl_step
            self.cl_ratio = min(self.cl_ratio,1.0)
            self.clear_cl_ep_success_dq()
        elif success_rate < self.cl_lower_cond:
            self.cl_ratio -= self.cl_step
            self.cl_ratio = max(self.cl_ratio,0.0)
            self.clear_cl_ep_success_dq()
    
    def clear_cl_ep_success_dq(self):
        for _ in range(self.cl_ep_success_dq.maxlen):
            self.cl_ep_success_dq.append(0.0)
         
    




     