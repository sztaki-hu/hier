import time
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from abc import ABC, abstractmethod 

REWARD_TYPES = ['sparse','state_change_bonus']

class TaskEnvBase(ABC):

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
    
    @abstractmethod
    def reset(self) -> np.ndarray:
        pass

    @abstractmethod
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        pass

    @abstractmethod
    def random_sample(self) -> np.ndarray:
        pass

    @abstractmethod
    def shuttdown(self) -> None:
        pass
    
    # ISE  ###################################################
    @abstractmethod
    def get_init_ranges(self) -> Dict:
        pass

    @abstractmethod
    def get_obs(self) -> np.ndarray: 
        pass

    # @abstractmethod
    # def save_state(self):
    #     pass

    @abstractmethod
    def load_state(self, 
                   robot_joints: Optional[np.ndarray], 
                   desired_goal: np.ndarray, 
                   object_position: Optional[np.ndarray] = None
                   ) -> None:
        pass

    # HER ###################################################

    @abstractmethod
    def is_diff_state(self, 
                      o_start: np.ndarray, 
                      o_end: np.ndarray, 
                      dim: int = 2, 
                      threshold: float = 0.01
                      ) -> bool:
        pass

    @abstractmethod
    def get_achieved_goal_from_obs(self, o: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def get_desired_goal_from_obs(self, o: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def change_goal_in_obs(self, o: np.ndarray, goal: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def her_get_reward_and_done(self, o: np.ndarray) -> Tuple[float, bool]:
        pass
    
        





    
  

    
  
                 

