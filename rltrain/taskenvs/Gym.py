import numpy as np
import gymnasium as gym
import os
from typing import Dict, List, Tuple, Any, Optional

from rltrain.taskenvs.TaskEnvBase import TaskEnvBase
from gymnasium import Env

class Gym(TaskEnvBase):

    def __init__(self,config: Dict, config_framework: Dict) -> None:
        super(Gym, self).__init__(config,config_framework)
        
        if self.task_name not in config_framework['task_list']['gym']: 
            raise ValueError("[TaskEnv Gym]: task_name: '" + str(self.task_name) + "' must be in : " + str(config_framework['task_list']['gym']))

         # Create taskenv
        self.env = gym.make(self.task_name) if self.headless == True else gym.make(self.task_name, render_mode="human") 
        self.env._max_episode_steps = int(float(self.max_ep_len)) # type: ignore

        self.reset()  
    
    def reset(self) -> np.ndarray:
        o_dict, _ = self.env.reset()
        o = np.concatenate((o_dict['observation'], o_dict['desired_goal']))
        self.ep_o_start = o.copy()
        self.obs = o.copy()
        return o   

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:

        o_dict, r, terminated, truncated, info = self.env.step(action)

        o = np.concatenate((o_dict['observation'], o_dict['desired_goal']))
        self.obs = o.copy() 
       
        r_float = float(r)
        r_float -= 1
        if self.reward_shaping_type == 'state_change_bonus':
            raise ValueError("[TaskEnv Gym]: state_change_bonus is not implemented")
        r_float = r_float * self.reward_scalor

        info['is_success'] = info.pop('success')

        return o, r_float, terminated, truncated, info 

    def random_sample(self) -> np.ndarray:
        return self.env.action_space.sample() 

    def shuttdown(self) -> None:
        self.reset()
        self.env.close()
    
    # ISE  not-implemented ###################################################
    def get_init_ranges(self) -> Dict:
        return {}

    def get_obs(self) -> np.ndarray: 
        return self.obs
    
    def load_state(self, 
                desired_goal: np.ndarray, 
                ) -> None:
        self.env.unwrapped.goal = desired_goal # type: ignore
        self.env.unwrapped.update_target_site_pos() # type: ignore

    # HER ###################################################

    def is_diff_state(self, 
                    o_start: np.ndarray, 
                    o_end: np.ndarray, 
                    dim: int = 2, 
                    threshold: float = 0.01
                    ) -> bool:
        obj_start_pos = self.get_achieved_goal_from_obs(o_start)
        obj_end_pos = self.get_achieved_goal_from_obs(o_end)

        distance =  np.linalg.norm(obj_start_pos[:dim] - obj_end_pos[:dim], axis=-1)
    
        return bool(np.array(distance > threshold, dtype=np.float32))

    def get_achieved_goal_from_obs(self, o: np.ndarray) -> np.ndarray:
        if self.task_name in ['PointMaze_UMaze-v3']:
            o2 = o.copy()
            return o2[:2]
        else:
            raise ValueError("[TaskEnv Gym]: get_achieved_goal() for " + self.task_name + " is not defined.")

    def get_desired_goal_from_obs(self, o: np.ndarray) -> np.ndarray:
        if self.task_name in ['PointMaze_UMaze-v3']:
            return o[-2:].copy()
        else:
            raise ValueError("[TaskEnv Gym]: get_desired_goal() for " + self.task_name + " is not defined.")


    def change_goal_in_obs(self, o: np.ndarray, goal: np.ndarray) -> np.ndarray:
        o2 = o.copy()
        if self.task_name in ['PointMaze_UMaze-v3']:
            o2[-2:] = goal.copy()
            return o2
        else:
            raise ValueError("[TaskEnv Gym]: change_goal_in_obs() for " + self.task_name + " is not defined.")

    def her_get_reward_and_done(self, o: np.ndarray) -> Tuple[float, bool]:
        desired_goal = self.get_desired_goal_from_obs(o)
        achieved_goal = self.get_achieved_goal_from_obs(o)

        d = np.linalg.norm(achieved_goal - desired_goal)
        r = 0.0 if d <= 0.45 else -1.0
        return r,d
    


   

    

    
  
                 

