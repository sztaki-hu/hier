import numpy as np
import random
import math

from rltrain.algos.cl_teachers.CL import CL

class PredefinedCL(CL):
   
    def __init__(self, config, env, replay_buffer):
        super(PredefinedCL, self).__init__(config, env, replay_buffer)

        # PREDEFINED
        self.cl_pacing_profile = self.config['trainer']['cl']['predefined']['pacing_profile']
        self.cl_pacing_sat = self.config['trainer']['cl']['predefined']['pacing_sat']
        self.t_ratio = 0
 
    def update_ratio(self,t):
        if self.cl_pacing_profile == "linear":
            self.t_ratio = min(1.0,t / float(self.total_timesteps * self.cl_pacing_sat))
        elif self.cl_pacing_profile == "sqrt":
            self.t_ratio = min(1.0,math.sqrt(t / float(self.total_timesteps * self.cl_pacing_sat)))
        elif self.cl_pacing_profile == "quad":
            self.t_ratio = min(1.0,math.pow(t / float(self.total_timesteps * self.cl_pacing_sat),2))
    




     