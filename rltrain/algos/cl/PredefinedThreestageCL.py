import numpy as np
import random
import math

from rltrain.algos.cl.CL import CL

PACING_PROFILES = ['linear','sqrt','quad']

class PredefinedThreestageCL(CL):
   
    def __init__(self, config, env, replay_buffer):
        super(PredefinedThreestageCL, self).__init__(config, env, replay_buffer)

        # PREDEFINED 3-STAGE
        self.cl_pacing_profile = self.config['trainer']['cl']['predefinedthreestage']['pacing_profile']
        self.cl_change_stage12 = self.config['trainer']['cl']['predefinedthreestage']['change_stage12']
        self.cl_change_stage23 = self.config['trainer']['cl']['predefinedthreestage']['change_stage23']
        self.cl_stage1_pacing_sat = self.config['trainer']['cl']['predefinedthreestage']['stage1']['pacing_sat']
        self.cl_stage2_pacing_sat = self.config['trainer']['cl']['predefinedthreestage']['stage2']['pacing_sat']
        self.cl_stage3_pacing_sat = self.config['trainer']['cl']['predefinedthreestage']['stage3']['pacing_sat']
        self.cl_ratio = 0
        self.cl_ratio_discard = 0
        self.store_rollout_success_rate = False

        self.cl_obj_ratio = 0
        self.cl_goal_ratio = 0
        self.cl_obj_ratio_discard  = 0
        self.cl_goal_ratio_discard  = 0

        print(self.cl_pacing_profile )
        assert self.cl_pacing_profile in PACING_PROFILES
 
    def update_cl(self,t):
        t = float(t) / self.total_timesteps
        if t < self.cl_change_stage12:  # stage 1
            # CL obj ratio
            self.cl_obj_ratio = 0
            # CL goal ratio
            self.cl_goal_ratio = self.get_ratio(t,self.cl_stage1_pacing_sat)
            self.cl_goal_ratio_discard = max(0.0, self.cl_goal_ratio - self.cl_ratio_discard_lag)
            # Update cl ratio for logging
            self.cl_ratio = self.cl_goal_ratio
        elif t >= self.cl_change_stage12 and t < self.cl_change_stage23:  # stage 2
            # CL obj ratio 
            self.cl_obj_ratio = self.get_ratio(t - self.cl_change_stage12, self.cl_stage2_pacing_sat - self.cl_change_stage12)
            self.cl_obj_ratio_discard = max(0.0, self.cl_goal_ratio - self.cl_ratio_discard_lag)
            # CL goal ratio 
            self.cl_goal_ratio = 0
            # Update cl ratio for logging
            self.cl_ratio = self.cl_obj_ratio
        elif t >= self.cl_change_stage23: # stage 3
            # CL obj ratio 
            self.cl_obj_ratio = self.get_ratio(t - self.cl_change_stage23, self.cl_stage3_pacing_sat - self.cl_change_stage23)
            self.cl_obj_ratio_discard = max(0.0, self.cl_goal_ratio - self.cl_ratio_discard_lag)
            # CL goal ratio 
            self.cl_goal_ratio = self.get_ratio(t - self.cl_change_stage23, self.cl_stage3_pacing_sat - self.cl_change_stage23)
            self.cl_goal_ratio_discard = max(0.0, self.cl_goal_ratio - self.cl_ratio_discard_lag)
            # Update cl ratio for logging
            self.cl_ratio = self.cl_goal_ratio
    
    def get_range(self,placeholder):        
        goal_low = self.goal_range_center - self.goal_range_half * self.cl_goal_ratio
        goal_high = self.goal_range_center + self.goal_range_half * self.cl_goal_ratio
        obj_low = self.obj_range_center - self.obj_range_half * self.cl_obj_ratio
        obj_high = self.obj_range_center + self.obj_range_half * self.cl_obj_ratio
        return goal_low, goal_high, obj_low, obj_high
    
    def get_ratio(self, t, pacing_sat):
        if self.cl_pacing_profile == "linear":
            return min(1.0,t /  pacing_sat)
        elif self.cl_pacing_profile == "sqrt":
            return min(1.0,math.sqrt(t / pacing_sat))
        elif self.cl_pacing_profile == "quad":
            return min(1.0,math.pow(t / pacing_sat,2))
        




     