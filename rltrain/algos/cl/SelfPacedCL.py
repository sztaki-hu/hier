import numpy as np
import collections

from rltrain.algos.cl.CL import CL

class SelfPacedCL(CL):

    def __init__(self, config, env, replay_buffer):
        super(SelfPacedCL, self).__init__(config, env, replay_buffer)

        # SELFPACED
        self.cl_conv_cond = self.config['trainer']['cl']['selfpaced']['conv_cond']
        self.cl_step = self.config['trainer']['cl']['selfpaced']['step']
        self.cl_dequeu_maxlen = config['trainer']['cl']['selfpaced']['window_size']
        
        self.store_rollout_success_rate = True
        self.cl_rollout_success_dq = collections.deque(maxlen=self.cl_dequeu_maxlen)
   
    def update_cl(self,t):
        if len(self.cl_rollout_success_dq) == self.cl_dequeu_maxlen:          
            success_rate = np.mean(self.cl_rollout_success_dq)
            if success_rate > self.cl_conv_cond:
                self.cl_ratio += self.cl_step
                self.cl_ratio = min(self.cl_ratio,1.0)
                self.cl_rollout_success_dq.clear()
                self.cl_ratio_discard = max(0.0, self.cl_ratio - self.cl_ratio_discard_lag)

                self.copy_cl_ratios_to_obj_and_goal()
    

         
    




     