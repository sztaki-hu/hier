import numpy as np
import gymnasium as gym
import os
from typing import Dict, List, Tuple, Any

from rltrain.taskenvs.TaskEnvBase import TaskEnvBase
from gymnasium import Env

class Gym(TaskEnvBase):

    def __init__(self,config: Dict, config_framework: Dict) -> None:
        super(Gym, self).__init__(config, config_framework)

        # Create taskenv
        self.env = gym.make(self.task_name) if self.headless == True else gym.make(self.task_name, render_mode="human") 
        self.env._max_episode_steps = int(float(self.max_ep_len)) # type: ignore

        assert config['buffer']['her']['goal_selection_strategy'] == 'noher'
        assert config['trainer']['ise']['type'] == 'max'

        self.reset() 
    
    def reset(self) -> np.ndarray:
        o, _ = self.env.reset()
        self.t = 0
        self.obs = o.copy()
        return o     
   
   
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:

        o, r, terminated, truncated, info = self.env.step(action)
        
        self.t += 1
        info['is_success'] = True if self.t == self.max_ep_len else False
        
        r = r * self.reward_scalor 

        self.obs = o.copy()

        return o, r, terminated, truncated, info

    def shuttdown(self) -> None:
        self.reset()
        self.env.close()
    
    def get_obs(self) -> np.ndarray:
        return self.obs
    


   

    

    
  
                 

