import numpy as np
import random
import math

from typing import Dict, List, Tuple, Union, Optional
from rltrain.taskenvs.GymPanda import GymPanda

from rltrain.algos.ise.InitialStateEntropy import InitialStateEntropy

class MinISE(InitialStateEntropy):

    def __init__(self, config: Dict, taskenv: GymPanda) -> None:
        super(MinISE, self).__init__(config, taskenv)

        self.c = 0
        self.c_obj = 0
        self.c_goal = 0
   
    def reset_env(self,t: int) -> np.ndarray:
        self.taskenv.reset()
        return self.taskenv.get_obs()
    




     