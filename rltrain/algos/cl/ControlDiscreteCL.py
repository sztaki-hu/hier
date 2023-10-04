import numpy as np
import collections
import math

from rltrain.algos.cl.CL import CL

TARGET_PROFILES = ['const','const_sin','linear','sqrt','quad']

class ControlDiscreteCL(CL):

    def __init__(self, config, env, replay_buffer):
        super(ControlDiscreteCL, self).__init__(config, env, replay_buffer)

        # Control
        self.cl_target_profile = self.config['trainer']['cl']['controldiscrete']['target_profile']
        self.cl_target_sat = self.config['trainer']['cl']['controldiscrete']['target_sat']
        self.cl_target_sat_value = self.config['trainer']['cl']['controldiscrete']['target_sat_value']
        self.cl_step = self.config['trainer']['cl']['controldiscrete']['step']
        self.cl_dequeu_maxlen = config['trainer']['cl']['controldiscrete']['window_size']
        self.cl_ratio = 0 
        self.cl_ratio_discard = 0
        self.cl_rollout_success_dq = collections.deque(maxlen=self.cl_dequeu_maxlen)
        self.store_rollout_success_rate = True

        self.divide_linear = float(self.total_timesteps * self.cl_target_sat / self.cl_target_sat_value)
        self.divide_sqrt = float(self.total_timesteps * self.cl_target_sat / math.pow(self.cl_target_sat_value,2))
        self.divide_quad = float(self.total_timesteps * self.cl_target_sat / math.sqrt(self.cl_target_sat_value))

        # const_sin 
        self.const_sin_amp = config['trainer']['cl']['controldiscrete']['const_sin']['amp']
        self.const_sin_freq_divide = config['trainer']['cl']['controldiscrete']['const_sin']['freq_divide']

        assert self.cl_target_profile in TARGET_PROFILES
   
    def update_cl(self,t):  
        success_rate = np.mean(self.cl_rollout_success_dq) if t > 0 else 0
        if self.get_target(t) > success_rate: 
            self.cl_ratio -= self.cl_step
            self.cl_ratio = max(self.cl_ratio,0.0)
        else:
            self.cl_ratio += self.cl_step
            self.cl_ratio = min(self.cl_ratio,1.0)
    
    def get_target(self,t):
        if self.cl_target_profile == "const":
            return self.cl_target_sat_value
        elif self.cl_target_profile == "const_sin":
            return self.cl_target_sat_value + self.const_sin_amp * math.sin(t / (self.total_timesteps / self.const_sin_freq_divide))
        elif self.cl_target_profile == "linear":
            return min(self.cl_target_sat_value,t / self.divide_linear)
        elif self.cl_target_profile == "sqrt":
            return min(self.cl_target_sat_value,math.sqrt(t / self.divide_sqrt))
        elif self.cl_target_profile == "quad":
            return min(self.cl_target_sat_value,math.pow((t / self.divide_quad),2))
        self.cl_ratio_discard = max(0.0, self.cl_ratio - self.cl_ratio_discard_lag)

           
    




     