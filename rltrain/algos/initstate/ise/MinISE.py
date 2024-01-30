import numpy as np
import random
import math

from typing import Dict, List, Tuple, Union, Optional
from rltrain.taskenvs.GymPanda import GymPanda
from rltrain.taskenvs.GymMaze import GymMaze
from rltrain.taskenvs.GymFetch import GymFetch

from rltrain.algos.initstate.ise.InitStateEntropy import InitStateEntropy

class MinISE(InitStateEntropy):

    def __init__(self,
                 config: Dict,
                 taskenv: Union[GymPanda, GymMaze, GymFetch]
                 ) -> None:
        super(MinISE, self).__init__(config, taskenv)

        self.c = 0
        self.c_obj = 0
        self.c_goal = 0
   
    def reset_env(self,t: int) -> np.ndarray:
        self.taskenv.reset()
        return self.taskenv.get_obs()
    




     