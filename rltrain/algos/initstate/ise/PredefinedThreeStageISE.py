import numpy as np
import random
import math

from typing import Dict, List, Tuple, Union, Optional

from rltrain.taskenvs.GymPanda import GymPanda
from rltrain.algos.initstate.ise.InitStateEntropy import InitStateEntropy

PACING_PROFILES = ['linear','sqrt','quad']

class PredefinedThreeStageISE(InitStateEntropy):
   
    def __init__(self, config: Dict, taskenv: GymPanda) -> None:
        super(PredefinedThreeStageISE, self).__init__(config, taskenv)

        # PREDEFINED 3-STAGE
        self.profile = self.config['trainer']['init_state']['ise']['predefined3stage']['profile']
        self.change_stage12 = self.config['trainer']['init_state']['ise']['predefined3stage']['change_stage12']
        self.change_stage23 = self.config['trainer']['init_state']['ise']['predefined3stage']['change_stage23']
        self.stage1_saturation_t= self.config['trainer']['init_state']['ise']['predefined3stage']['stage1']['saturation_t']
        self.stage2_saturation_t = self.config['trainer']['init_state']['ise']['predefined3stage']['stage2']['saturation_t']
        self.stage3_saturation_t = self.config['trainer']['init_state']['ise']['predefined3stage']['stage3']['saturation_t']
        
        print(self.profile)
        assert self.profile in PACING_PROFILES
 
    def update_c(self, t: int) -> None:
        t_rel = float(t) / self.total_timesteps
        if t_rel < self.change_stage12:  # stage 1
            # ISE obj ratio
            self.c_obj = 0
            # ISE goal ratio
            self.c_goal = self.get_ratio(t_rel,self.stage1_saturation_t)
            self.c_goal_discard = max(0.0, self.c_goal - self.c_discard_lag)
            # Update ise ratio for logging
            self.c = self.c_goal
        elif t_rel >= self.change_stage12 and t_rel < self.change_stage23:  # stage 2
            # ISE obj ratio 
            self.c_obj = self.get_ratio(t_rel - self.change_stage12, self.stage2_saturation_t - self.change_stage12)
            self.c_obj_discard = max(0.0, self.c_goal - self.c_discard_lag)
            # ISE goal ratio 
            self.c_goal = 0
            # Update ise ratio for logging
            self.c = self.c_obj
        elif t_rel >= self.change_stage23: # stage 3
            # ISE obj ratio 
            self.c_obj = self.get_ratio(t_rel - self.change_stage23, self.stage3_saturation_t - self.change_stage23)
            self.c_obj_discard = max(0.0, self.c_goal - self.c_discard_lag)
            # ISE goal ratio 
            self.c_goal = self.get_ratio(t_rel - self.change_stage23, self.stage3_saturation_t - self.change_stage23)
            self.c_goal_discard = max(0.0, self.c_goal - self.c_discard_lag)
            # Update ise ratio for logging
            self.c = self.c_goal
    
    def get_ratio(self, t_rel: float, pacing_sat: float) -> float:
        if self.profile == "linear":
            return min(1.0,t_rel /  pacing_sat)
        elif self.profile == "sqrt":
            return min(1.0,math.sqrt(t_rel / pacing_sat))
        elif self.profile == "quad":
            return min(1.0,math.pow(t_rel / pacing_sat,2))
        else:
            raise ValueError("[ISE]: profile: '" + str(self.profile) + "' must be in : " + str(PACING_PROFILES))
   
        




     