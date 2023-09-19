import numpy as np
import random
import math

from rltrain.algos.cl_teachers.CL import CL

class PredefinedCL(CL):
 
   def update_ratio(self,t,ep_success_dq):
        if self.cl_pacing_profile == "linear":
            self.t_ratio = min(1.0,t / float(self.total_timesteps * self.cl_pacing_sat))
        elif self.cl_pacing_profile == "sqrt":
            self.t_ratio = min(1.0,math.sqrt(t / float(self.total_timesteps * self.cl_pacing_sat)))
        elif self.cl_pacing_profile == "quad":
            self.t_ratio = min(1.0,math.pow(t / float(self.total_timesteps * self.cl_pacing_sat),2))
    




     