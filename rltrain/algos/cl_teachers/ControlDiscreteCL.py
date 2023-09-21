import numpy as np
import collections
import math

from rltrain.algos.cl_teachers.CL import CL

class ControlDiscreteCL(CL):

    def __init__(self, config, env, replay_buffer):
        super(ControlDiscreteCL, self).__init__(config, env, replay_buffer)

        # Control
        self.cl_target = self.config['trainer']['cl']['controldiscrete']['target']
        self.cl_step = self.config['trainer']['cl']['controldiscrete']['step']
        self.cl_ratio = 0 
        self.cl_ep_success_dq = collections.deque(maxlen=config['trainer']['cl']['selfpaced']['window_size'])
   
    def update_ratio(self,t):
        success_rate = np.mean(self.cl_ep_success_dq) if t > 0 else 0
        delta = self.cl_target - success_rate
        if self.cl_ratio < 1.0:
            if delta > 0: # target > real
                self.cl_ratio -= self.cl_step
                self.cl_ratio = max(self.cl_ratio,0.0)
            else:
                self.cl_ratio += self.cl_step
                self.cl_ratio = min(self.cl_ratio,1.0)
           
    




     