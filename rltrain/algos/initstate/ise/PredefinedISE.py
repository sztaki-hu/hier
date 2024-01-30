import numpy as np
import random
import math

from typing import Dict, List, Tuple, Union, Optional

from rltrain.taskenvs.GymPanda import GymPanda
from rltrain.taskenvs.GymMaze import GymMaze
from rltrain.taskenvs.GymFetch import GymFetch
from rltrain.algos.initstate.ise.InitStateEntropy import InitStateEntropy

PACING_PROFILES = ['linear','sqrt','quad']

class PredefinedISE(InitStateEntropy):
   
    def __init__(self,
                 config: Dict,
                 taskenv: Union[GymPanda, GymMaze, GymFetch]
                 ) -> None:
        super(PredefinedISE, self).__init__(config, taskenv)

        # PREDEFINED
        self.profile = self.config['trainer']['init_state']['ise']['predefined']['profile']
        self.saturation_t = self.config['trainer']['init_state']['ise']['predefined']['saturation_t']

        assert self.profile in PACING_PROFILES
 
    def update_c(self,t: int) -> None:
        if self.profile == "linear":
            self.c = min(1.0,t / float(self.total_timesteps * self.saturation_t))
        elif self.profile == "sqrt":
            self.c = min(1.0,math.sqrt(t / float(self.total_timesteps * self.saturation_t)))
        elif self.profile == "quad":
            self.c = min(1.0,math.pow(t / float(self.total_timesteps * self.saturation_t),2))
        self.c_discard = max(0.0, self.c - self.c_discard_lag)

        self.copy_c_to_obj_and_goal()
    




     