import numpy as np
import random
import math

from typing import Dict, List, Tuple, Union, Optional

from rltrain.taskenvs.GymPanda import GymPanda

from rltrain.algos.initstate.ise.InitStateEntropy import InitStateEntropy

PACING_PROFILES = ['linear','sqrt','quad']

class PredefinedTwoStageISE(InitStateEntropy):
   
    def __init__(self, config: Dict, taskenv: GymPanda) -> None:
        super(PredefinedTwoStageISE, self).__init__(config, taskenv)

        # PREDEFINED 2-STAGE
        self.profile = self.config['trainer']['init_state']['ise']['predefined2stage']['profile']
        self.change_stage = self.config['trainer']['init_state']['ise']['predefined2stage']['change_stage']
        self.stage1_saturation_t = self.config['trainer']['init_state']['ise']['predefined2stage']['stage1']['saturation_t']
        self.stage2_saturation_t = self.config['trainer']['init_state']['ise']['predefined2stage']['stage2']['saturation_t']

        print(self.profile)
        assert self.profile in PACING_PROFILES
 
    def update_c(self, t: int) -> None:
        self.update_c_goal(t)
        self.update_c_obj(t)

    def update_c_goal(self, t: int) -> None:
        t_rel = float(t) / self.total_timesteps
        if t_rel < self.change_stage:
            if self.profile == "linear":
                self.c_goal = min(1.0,t_rel /  self.stage1_saturation_t)
            elif self.profile == "sqrt":
                self.c_goal = min(1.0,math.sqrt(t_rel / self.stage1_saturation_t))
            elif self.profile == "quad":
                self.c_goal = min(1.0,math.pow(t_rel / self.stage1_saturation_t,2))
            self.c_goal_discard = max(0.0, self.c_goal - self.c_discard_lag)
            self.c = self.c_goal

    def update_c_obj(self, t: int) -> None:
        t_rel = float(t) / self.total_timesteps
        if t_rel >= self.change_stage:
            if self.profile == "linear":
                self.c_obj = min(1.0,(t_rel - self.change_stage) / (self.stage2_saturation_t - self.change_stage))
            elif self.profile == "sqrt":
                self.c_obj = min(1.0,math.sqrt((t_rel - self.change_stage) / (self.stage2_saturation_t - self.change_stage)))
            elif self.profile == "quad":
                self.c_obj = min(1.0,math.pow((t_rel - self.change_stage) / (self.stage2_saturation_t - self.change_stage),2))
            self.c_obj_discard = max(0.0, self.c_obj - self.c_discard_lag)
            self.c = self.c_obj
    
 




     