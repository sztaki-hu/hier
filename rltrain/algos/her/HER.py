import numpy as np
import random
from typing import Dict, List, Tuple, Union
Transition = Tuple[np.ndarray,np.ndarray,float,np.ndarray,bool]

from rltrain.taskenvs.GymPanda import GymPanda
from rltrain.taskenvs.GymMaze import GymMaze
from rltrain.taskenvs.GymFetch import GymFetch
from rltrain.buffers.replay import ReplayBuffer 
from rltrain.buffers.prioritized_replay import PrioritizedReplay


class HER:
    def __init__(self, 
                 config: Dict, 
                 config_framework: Dict,
                 taskenv: Union[GymPanda, GymMaze, GymFetch], 
                 replay_buffer: Union[ReplayBuffer, PrioritizedReplay]
                 ) -> None:
        self.config = config
        self.config_framework = config_framework
        self.taskenv = taskenv
        self.replay_buffer = replay_buffer

        # HER
        self.her_goal_selection_strategy = config['buffer']['her']['goal_selection_strategy']
        self.her_active = False if self.her_goal_selection_strategy == "noher" else True
        self.n_sampled_goal = config['buffer']['her']['n_sampled_goal']
        self.state_check = config['buffer']['her']['state_check']
  
        if self.her_goal_selection_strategy not in config_framework['her']['mode_list']:
            raise ValueError("[HER]: her_goal_selection_strategy: '" + str(self.her_goal_selection_strategy) + "' must be in : " + str(config_framework['her']['strategy_list']))

    def get_new_goals(self, episode: List[Transition], ep_t: int) -> List[np.ndarray]:
        if self.her_goal_selection_strategy == 'final':
            new_goals = []
            _, _, _, o2, _ = episode[-1]
            for _ in range(self.n_sampled_goal):
                new_goals.append(self.taskenv.get_achieved_goal_from_obs(o2))
            return new_goals
        elif self.her_goal_selection_strategy == 'final_valid': #for slide
            new_goals = []
            for i in range(len(episode)):
                _, _, _, o2, _ = episode[i]
                if o2[5] < 0.35: break
            _, _, _, o2, _ = episode[i]
            for _ in range(self.n_sampled_goal):
                new_goals.append(self.taskenv.get_achieved_goal_from_obs(o2))
            return new_goals
        elif self.her_goal_selection_strategy == 'future' or self.her_goal_selection_strategy == 'future_once':
            new_goals = []
            for _ in range(self.n_sampled_goal):
                rand_future_transition = random.randint(ep_t, len(episode)-1)
                _, _, _, o2, _ = episode[rand_future_transition]
                new_goals.append(self.taskenv.get_achieved_goal_from_obs(o2))
            return new_goals
        elif self.her_goal_selection_strategy == 'near':
            new_goals = []
            for _ in range(self.n_sampled_goal):
                rand_future_transition = random.randint(ep_t, min(len(episode)-1,ep_t+5))
                _, _, _, o2, _ = episode[rand_future_transition]
                new_goals.append(self.taskenv.get_achieved_goal_from_obs(o2))
            return new_goals
        elif self.her_goal_selection_strategy == 'next':
            new_goals = []
            for _ in range(self.n_sampled_goal):
                _, _, _, o2, _ = episode[ep_t]
                new_goals.append(self.taskenv.get_achieved_goal_from_obs(o2))
            return new_goals
        else:
            raise ValueError("[HER]: her_goal_selection_strategy: '" + str(self.her_goal_selection_strategy) + "' must be in : " + str(self.config_framework['her']['mode_list']))

    
    def add_virtial_experience(self, episode: List[Transition]) -> bool:

        state_changed = self.taskenv.is_diff_state(episode[0][0], episode[-1][0])

        if state_changed or self.state_check == False:
            if self.her_goal_selection_strategy in ['final','final_valid','future_once']:
                new_goals = self.get_new_goals(episode,0)
                for (o, a, r, o2, d) in episode:                  
                    for new_goal in new_goals:
                        o_new = self.taskenv.change_goal_in_obs(o, new_goal)
                        o2_new = self.taskenv.change_goal_in_obs(o2, new_goal)
                        r_new, d_new = self.taskenv.her_get_reward_and_done(o2_new) 
                        self.replay_buffer.store(o_new, a, r_new, o2_new, d_new)
            else:
                ep_t = 0
                for (o, a, r, o2, d) in episode:
                    new_goals = self.get_new_goals(episode,ep_t)
                    for new_goal in new_goals:
                        o_new = self.taskenv.change_goal_in_obs(o, new_goal)
                        o2_new = self.taskenv.change_goal_in_obs(o2, new_goal)
                        r_new, d_new = self.taskenv.her_get_reward_and_done(o2_new) 
                        self.replay_buffer.store(o_new, a, r_new, o2_new, d_new)
                    ep_t += 1
        
        return state_changed