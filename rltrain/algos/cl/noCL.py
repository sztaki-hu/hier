import numpy as np
import random
import math

from typing import Dict, List, Tuple, Union, Optional

from rltrain.taskenvs.GymPanda import GymPanda
from rltrain.algos.cl.CL import CL


class noCL(CL):

    def __init__(self, config: Dict, taskenv: GymPanda) -> None:
        super(noCL, self).__init__(config, taskenv)

        self.c = 1
        self.c_obj = 1
        self.c_goal = 1
   
    def reset_env(self, t: int) -> np.ndarray:
        self.taskenv.reset()
        return self.taskenv.get_obs()
    




     