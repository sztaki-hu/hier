import numpy as np
import random
import collections

from typing import Dict, List, Tuple, Union, Optional
from rltrain.taskenvs.GymPanda import GymPanda
from rltrain.algos.initstate.InitState import InitState

from abc import abstractmethod 

class InitStateDiscEntropy(InitState):
    def __init__(self, config: Dict, taskenv: GymPanda) -> None:

        # INIT CONFIG
        self.config = config
        self.taskenv = taskenv

        # TASK
        self.task_name = self.config['environment']['task']['name']

        # INIT
        self.total_timesteps = float(config['trainer']['total_timesteps'])

        self.c = 0
        self.c_obj = 0
        self.c_goal = 0

        self.g = config['trainer']['init_state']['isedisc']['g']
        self.r = config['trainer']['init_state']['isedisc']['r']

        self.store_rollout_success_rate = False
        self.store_eval_success_rate = False
        self.eval_success_dq = collections.deque()
        self.rollout_success_dq = collections.deque() 
    
    def reset_env(self, t: int) -> np.ndarray:

        self.update_c(t)

        options = self.get_options()
        self.taskenv.reset(options = options)
        
        return self.taskenv.get_obs()

    @abstractmethod
    def update_c(self,t: int) -> None:
        pass

    def get_options(self) -> Dict:
        
        max_index = max(1, int(self.c*len(self.g)))
        g_available = self.g[:max_index]
        goal = random.choice(g_available)

        max_index = max(1, int(self.c*len(self.r)))
        r_available = self.r[:max_index]
        robot = random.choice(r_available)

        options = {
            "goal_cell": np.array([goal[0],goal[1]]),
            "reset_cell": np.array([robot[0],robot[1]]),
            }
        
        return options
    
    def append_rollout_success_dq(self, ep_succes: float) -> None:
        if self.store_rollout_success_rate: self.rollout_success_dq.append(ep_succes)
    
    def append_eval_success_dq(self, ep_succes: float) -> None:
        if self.store_rollout_success_rate: self.eval_success_dq.append(ep_succes)


   
    



    




     