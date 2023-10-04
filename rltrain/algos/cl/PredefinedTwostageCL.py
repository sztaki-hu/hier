import numpy as np
import random
import math

from rltrain.algos.cl.CL import CL

PACING_PROFILES = ['linear','sqrt','quad']

class PredefinedTwostageCL(CL):
   
    def __init__(self, config, env, replay_buffer):
        super(PredefinedTwostageCL, self).__init__(config, env, replay_buffer)

        # PREDEFINED 2-STAGE
        self.cl_pacing_profile = self.config['trainer']['cl']['predefinedtwostage']['pacing_profile']
        self.cl_change_stage = self.config['trainer']['cl']['predefinedtwostage']['change_stage']
        self.cl_stage1_pacing_sat = self.config['trainer']['cl']['predefinedtwostage']['stage1']['pacing_sat']
        self.cl_stage2_pacing_sat = self.config['trainer']['cl']['predefinedtwostage']['stage2']['pacing_sat']
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
        self.update_cl_goal_ratio(t)
        self.update_cl_obj_ratio(t)

    def update_cl_goal_ratio(self,t):
        t = float(t) / self.total_timesteps
        if t < self.cl_change_stage:
            if self.cl_pacing_profile == "linear":
                self.cl_goal_ratio = min(1.0,t /  self.cl_stage1_pacing_sat)
            elif self.cl_pacing_profile == "sqrt":
                self.cl_goal_ratio = min(1.0,math.sqrt(t / self.cl_stage1_pacing_sat))
            elif self.cl_pacing_profile == "quad":
                self.cl_goal_ratio = min(1.0,math.pow(t / self.cl_stage1_pacing_sat,2))
            self.cl_goal_ratio_discard = max(0.0, self.cl_goal_ratio - self.cl_ratio_discard_lag)
            self.cl_ratio = self.cl_goal_ratio

    def update_cl_obj_ratio(self,t):
        t = float(t) / self.total_timesteps
        if t >= self.cl_change_stage:
            if self.cl_pacing_profile == "linear":
                self.cl_obj_ratio = min(1.0,(t - self.cl_change_stage) / (self.cl_stage2_pacing_sat - self.cl_change_stage))
            elif self.cl_pacing_profile == "sqrt":
                self.cl_obj_ratio = min(1.0,math.sqrt((t - self.cl_change_stage) / (self.cl_stage2_pacing_sat - self.cl_change_stage)))
            elif self.cl_pacing_profile == "quad":
                self.cl_obj_ratio = min(1.0,math.pow((t - self.cl_change_stage) / (self.cl_stage2_pacing_sat - self.cl_change_stage),2))
            self.cl_obj_ratio_discard = max(0.0, self.cl_obj_ratio - self.cl_ratio_discard_lag)
            self.cl_ratio = self.cl_obj_ratio
    
    def get_range(self,placeholder):        
        goal_low = self.goal_range_center - self.goal_range_half * self.cl_goal_ratio
        goal_high = self.goal_range_center + self.goal_range_half * self.cl_goal_ratio
        obj_low = self.obj_range_center - self.obj_range_half * self.cl_obj_ratio
        obj_high = self.obj_range_center + self.obj_range_half * self.cl_obj_ratio
        return goal_low, goal_high, obj_low, obj_high




     