import numpy as np
import random
import collections

from typing import Dict, List, Tuple, Union, Optional
from abc import ABC, abstractmethod 

from rltrain.taskenvs.GymPanda import GymPanda
from rltrain.taskenvs.GymMaze import GymMaze
from rltrain.taskenvs.GymFetch import GymFetch

class InitState(ABC):
    def __init__(self, config: Dict, 
                 taskenv: Union[GymPanda, GymMaze, GymFetch]
                 ) -> None:

        # INIT CONFIG
        self.config = config
        self.taskenv = taskenv

        # TASK
        self.task_name = self.config['environment']['task']['name']

    @abstractmethod
    def reset_env(self, t: int) -> np.ndarray:
        pass

   

    



    




     