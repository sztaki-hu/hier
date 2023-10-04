import numpy as np
import random
import math

from rltrain.algos.cl.CL import CL

PACING_PROFILES = ['linear','sqrt','quad']

class PredefinedCL(CL):
   
    def __init__(self, config, env, replay_buffer):
        super(PredefinedCL, self).__init__(config, env, replay_buffer)

        # PREDEFINED
        self.cl_pacing_profile = self.config['trainer']['cl']['predefined']['pacing_profile']
        self.cl_pacing_sat = self.config['trainer']['cl']['predefined']['pacing_sat']
        self.cl_ratio = 0
        self.cl_ratio_discard = 0
        self.store_rollout_success_rate = False

        assert self.cl_pacing_profile in PACING_PROFILES
 
    def update_cl(self,t):
        if self.cl_pacing_profile == "linear":
            self.cl_ratio = min(1.0,t / float(self.total_timesteps * self.cl_pacing_sat))
        elif self.cl_pacing_profile == "sqrt":
            self.cl_ratio = min(1.0,math.sqrt(t / float(self.total_timesteps * self.cl_pacing_sat)))
        elif self.cl_pacing_profile == "quad":
            self.cl_ratio = min(1.0,math.pow(t / float(self.total_timesteps * self.cl_pacing_sat),2))
        self.cl_ratio_discard = max(0.0, self.cl_ratio - self.cl_ratio_discard_lag)
    




     