import time
import numpy as np
from typing import Dict, List, Tuple

REWARD_TYPES = ['sparse','state_change_bonus']

class TaskEnvBase:

    def __init__(self,config: Dict, config_framework: Dict) -> None:
        
        self.config = config
        self.config_framework = config_framework

        # General
        self.task_name = self.config['environment']['task']['name']
        self.headless = config['environment']['headless']
        self.max_ep_len = config['sampler']['max_ep_len']
        self.t = 0         
        self.obs = None  

        # Reward
        self.reward_shaping_type = config['environment']['reward']['reward_shaping_type']
        self.reward_scalor = config['environment']['reward']['reward_scalor']
        self.reward_bonus = config['environment']['reward']['reward_bonus']

        
        assert self.reward_shaping_type in REWARD_TYPES
    
    
    def init_state_valid(self, o: np.ndarray) -> bool:
        return True  

    def is_success(self) -> bool:
        return False       





    
  

    
  
                 

