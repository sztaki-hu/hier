import numpy as np
import random
import math

from rltrain.algos.cl_teachers.CL import CL

class SelfPacedCL(CL):
   
    def update_ratio(self,t,ep_success_dq):
        success_rate = np.mean(ep_success_dq) if t > 0 else 0
        if success_rate > self.cl_conv_cond:
            self.t_ratio += self.cl_step
            self.t_ratio = min(self.t_ratio,1.0)
         
    




     