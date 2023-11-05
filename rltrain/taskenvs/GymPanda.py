import numpy as np
import gymnasium as gym
import panda_gym
import time
from typing import Dict, List, Tuple, Any, Optional



from rltrain.taskenvs.TaskEnvBase import TaskEnvBase

class GymPanda(TaskEnvBase):

    def __init__(self,config: Dict, config_framework: Dict) -> None:
        super(GymPanda, self).__init__(config,config_framework)

        if self.task_name not in config_framework['task_list']['gympanda']: 
            raise ValueError("[TaskEnv GymPanda]: task_name: '" + str(self.task_name) + "' must be in : " + str(config_framework['task_list']['gympanda']))


        # Create taskenv
        self.env = gym.make(self.task_name) if self.headless == True else gym.make(self.task_name, render_mode="human") 
        self.env._max_episode_steps = int(float(self.max_ep_len)) # type: ignore
        
        # Init Obj Num and goal Num
        if self.task_name in ['PandaReach-v3','PandaReachDense-v3']:
            self.obj_num = 0
            self.goal_num = 1
        elif self.task_name in ['PandaStack-v3','PandaStackDense-v3']:
            self.obj_num = 2
            self.goal_num = 2
        else:
            self.obj_num = 1
            self.goal_num = 1
        
        self.reset() 
    
    def reset(self) -> np.ndarray:
        o_dict, _ = self.env.reset()
        o = np.concatenate((o_dict['observation'], o_dict['desired_goal']))
        self.ep_o_start = o.copy()
        return o   

    def save_state(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        robot_joints = np.array([self.env.robot.get_joint_angle(joint=i) for i in range(7)]) # type: ignore
        desired_goal = self.env.task.goal # type: ignore
        object_position = self.env.task.get_achieved_goal() # type: ignore

        return robot_joints,desired_goal,object_position

    def get_robot_joints(self) -> np.ndarray:
        return np.array([self.env.robot.get_joint_angle(joint=i) for i in range(7)]) # type: ignore
    
    def load_state(self, 
                   robot_joints: Optional[np.ndarray], 
                   desired_goal: np.ndarray, 
                   object_position: Optional[np.ndarray] = None
                   ) -> None:

        if robot_joints is not None: self.env.robot.set_joint_angles(robot_joints) # type: ignore

        if self.goal_num == 1:
            self.env.task.goal = desired_goal # type: ignore
            self.env.task.sim.set_base_pose("target", desired_goal, np.array([0.0, 0.0, 0.0, 1.0])) # type: ignore
        elif self.goal_num == 2:  
            self.env.task.goal = desired_goal # type: ignore
            self.env.task.sim.set_base_pose("target1", desired_goal[:3], np.array([0.0, 0.0, 0.0, 1.0])) # type: ignore
            self.env.task.sim.set_base_pose("target2", desired_goal[3:], np.array([0.0, 0.0, 0.0, 1.0])) # type: ignore
        else:
            raise ValueError("[TaskEnv GymPanda]: goal_num: " + str(self.goal_num) + " must be in : " + str([1,2]))

        if self.obj_num == 0:
            pass
        elif self.obj_num == 1:
            self.env.task.sim.set_base_pose("object", object_position, np.array([0.0, 0.0, 0.0, 1.0])) # type: ignore
        elif self.obj_num == 2:
            self.env.task.sim.set_base_pose("object1", object_position[:3], np.array([0.0, 0.0, 0.0, 1.0])) # type: ignore
            self.env.task.sim.set_base_pose("object2", object_position[3:], np.array([0.0, 0.0, 0.0, 1.0])) # type: ignore
        else:
            raise ValueError("[TaskEnv GymPanda]: obj_num: " + str(self.obj_num) + " must be in : " + str([0,1,2]))
      
    # def init_state_valid(self, o):
    #     if self.task_name == 'PandaPush-v3':
    #         o_goal = o[-3:]
    #         o_obj = o[6:9]  
    #         if np.allclose(o_goal, o_obj, rtol=0.0, atol=0.05, equal_nan=False):
    #             return False
        
    #     return True  
    
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:

        o_dict, r, terminated, truncated, info = self.env.step(action)

        o = np.concatenate((o_dict['observation'], o_dict['desired_goal']))
       
        r_float = float(r)
        if self.reward_shaping_type == 'state_change_bonus':
            r_float += self.get_reward_bonus(o)
        r_float = r_float * self.reward_scalor

        return o, r_float, terminated, truncated, info 
    
    def random_sample(self) -> np.ndarray:
        return self.env.action_space.sample() 


    def get_reward_bonus(self, o: np.ndarray) -> float:
        if self.task_name in ['PandaReach-v3','PandaReachDense-v3']:
            return 0.0 
        if self.task_name in ['PandaPush-v3','PandaPushDense-v3',
                              'PandaSlide-v3','PandaSlideDense-v3']:
            return self.reward_bonus if self.is_diff_state(self.ep_o_start,o) else 0.0
        elif self.task_name in ['PandaPickAndPlace-v3','PandaPickAndPlaceDense-v3']:
            return self.reward_bonus if self.get_achieved_goal_from_obs(o)[2] > 0.25 else 0.0 
        elif self.task_name in ['PandaStack-v3','PandaStackDense-v3']:
            return self.reward_bonus if self.get_achieved_goal_from_obs(o)[5] > 0.25 else 0.0 
        else:
            return 0.0
        
    
    # Curriculum Learning ##############################

    def get_init_ranges(self) -> Dict:
        dict = {}
        if self.task_name in ['PandaReach-v3','PandaReachDense-v3']:
            dict['obj_range_low'] = None
            dict['obj_range_high'] = None
            dict['object_size'] = None
        else:
            dict['obj_range_low'] = self.env.task.obj_range_low # type: ignore
            dict['obj_range_high'] = self.env.task.obj_range_high # type: ignore
            dict['object_size'] = self.env.task.object_size # type: ignore
        dict['goal_range_low'] = self.env.task.goal_range_low # type: ignore
        dict['goal_range_high'] = self.env.task.goal_range_high # type: ignore

        dict['obj_num'] = self.obj_num
        dict['goal_num'] = self.goal_num
        
        return dict
    
    def get_obs(self) -> np.ndarray:  
        robot_obs = self.env.robot.get_obs().astype(np.float32) # type: ignore # robot state
        task_obs = self.env.task.get_obs().astype(np.float32)  # type: ignore # object position, velococity, etc...
        observation = np.concatenate([robot_obs, task_obs])
        o = np.concatenate((observation, self.env.task.get_goal().astype(np.float32))) # type: ignore
        return o
    
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

    # def get_first_stable_state_index(self):
    #     if self.task_name in ['PandaReach-v3','PandaReachDense-v3']:
    #         return 0
    #     elif self.task_name in ['PandaPush-v3','PandaPushDense-v3']:
    #         return 0
    #     elif self.task_name in ['PandaSlide-v3','PandaSlideDense-v3']:
    #         return 5
        
    
    # HER ##############################################
    
    def get_achieved_goal_from_obs(self, o: np.ndarray) -> np.ndarray:
        if self.task_name in ['PandaReach-v3','PandaReachDense-v3']:
            o2 = o.copy()
            return o2[:3]
        elif self.task_name in ['PandaPush-v3','PandaPushDense-v3']:
            o2 = o.copy()
            return o2[6:9]
        elif self.task_name in ['PandaSlide-v3','PandaSlideDense-v3']:
            o2 = o.copy()
            return o2[6:9]
        elif self.task_name in ['PandaPickAndPlace-v3','PandaPickAndPlaceDense-v3']:
            o2 = o.copy()
            return o2[7:10]
        else: #self.task_name in ['PandaStack-v3','PandaStackDense-v3']:
            o2 = o.copy()
            return np.concatenate([o2[7:10],o2[19:22]])


    def get_desired_goal_from_obs(self, o: np.ndarray) -> np.ndarray:
        if self.task_name in ['PandaStack-v3','PandaStackDense-v3']:
            return o[-6:].copy()
        else:
            return o[-3:].copy()

    def change_goal_in_obs(self, o: np.ndarray, goal: np.ndarray) -> np.ndarray:
        o2 = o.copy()
        if self.task_name in ['PandaStack-v3','PandaStackDense-v3']:
            o2[-6:] = goal.copy()
            return o2
        else:
            o2[-3:] = goal.copy()
            return o2
    
    def her_get_reward_and_done(self, o: np.ndarray) -> Tuple[float, bool]:
        desired_goal = self.get_desired_goal_from_obs(o)
        achieved_goal = self.get_achieved_goal_from_obs(o)

        r = self.env.task.compute_reward(achieved_goal, desired_goal, {}) # type: ignore
        d = True if self.env.task.is_success(achieved_goal, desired_goal) else False # type: ignore
        return r,d
    

    def shuttdown(self) -> None:
        self.reset()
        self.env.close()
    



            

        
    
    
  
                 

