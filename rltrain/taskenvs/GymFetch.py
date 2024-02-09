import numpy as np
import gymnasium as gym
import os
from typing import Dict, List, Tuple, Any, Optional

from rltrain.taskenvs.TaskEnvBase import TaskEnvBase
from gymnasium import Env

class GymFetch(TaskEnvBase):

    def __init__(self,config: Dict, config_framework: Dict) -> None:
        super(GymFetch, self).__init__(config,config_framework)
        
        if self.task_name not in config_framework['task_list']['gymfetch']: 
            raise ValueError("[TaskEnv GymFetch]: task_name: '" + str(self.task_name) + "' must be in : " + str(config_framework['task_list']['gymfetch']))

         # Create taskenv
        if self.headless == True:
            self.env = gym.make(self.task_name, 
                max_episode_steps=int(float(self.max_ep_len)))
        else: 
            self.env = gym.make(self.task_name, 
                render_mode="human", 
                max_episode_steps=int(float(self.max_ep_len)))

        self.reset()  
    
    def obsdict2obsarray(self,o_dict: Dict) -> np.ndarray:
        return np.concatenate((o_dict['observation'], o_dict['desired_goal']))
       
    
    def reset(self, options:Dict = {}) -> np.ndarray:
        o_dict, _ = self.env.reset()

        o = self.obsdict2obsarray(o_dict)
        
        self.ep_o_start = o.copy()
        self.obs = o.copy()
        return o   

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:

        # Env step
        o_dict, r, terminated, truncated, info = self.env.step(action)

        # Chang obs data representation
        o = self.obsdict2obsarray(o_dict)
        self.obs = o.copy() 
       
        # Change reward 
        r_float = float(r)
        if self.reward_shaping_type == 'state_change_bonus':
            raise ValueError("[TaskEnv GymFetch]: state_change_bonus is not implemented")
        r_float = r_float * self.reward_scalor

        if info['is_success']: terminated = True 
        if self.task_name in ['FetchSlide-v2']:
            if o[5] < 0.4: terminated = True 

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
                robot_joints: Optional[np.ndarray], 
                desired_goal: np.ndarray, 
                object_position: Optional[np.ndarray] = None
                ) -> None:
        raise ValueError("[TaskEnv GymFetch]: load_state() for " + self.task_name + " is not defined.")

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
        if self.task_name in ['FetchReach-v2']:
            o2 = o.copy()
            return o2[:3]
        elif self.task_name in ['FetchPush-v2','FetchSlide-v2','FetchPickAndPlace-v2']:
            o2 = o.copy()
            return o2[3:6]
        else:
            raise ValueError("[TaskEnv GymFetch]: get_achieved_goal() for " + self.task_name + " is not defined.")

    def get_desired_goal_from_obs(self, o: np.ndarray) -> np.ndarray:
        if self.task_name in ['FetchReach-v2','FetchPush-v2','FetchSlide-v2','FetchPickAndPlace-v2']:
            return o[-3:].copy()
        else:
            raise ValueError("[TaskEnv GymFetch]: get_desired_goal() for " + self.task_name + " is not defined.")


    def change_goal_in_obs(self, o: np.ndarray, goal: np.ndarray) -> np.ndarray:
        o2 = o.copy()
        if self.task_name in ['FetchReach-v2','FetchPush-v2','FetchSlide-v2','FetchPickAndPlace-v2']:
            o2[-3:] = goal.copy()
            return o2
        else:
            raise ValueError("[TaskEnv GymFetch]: change_goal_in_obs() for " + self.task_name + " is not defined.")

    def her_get_reward_and_done(self, o: np.ndarray) -> Tuple[float, bool]:
        desired_goal = self.get_desired_goal_from_obs(o)
        achieved_goal = self.get_achieved_goal_from_obs(o)

        distance = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        r = 0.0 if distance <= 0.005 else -1.0
        return r,distance
    


   

    

    
  
                 

