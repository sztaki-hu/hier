import numpy as np
import random

class HER:
    def __init__(self,config, env, replay_buffer):
        self.config = config
        self.env = env
        self.replay_buffer = replay_buffer

        # HER
        self.her_goal_selection_strategy = config['buffer']['her']['goal_selection_strategy']
        self.her_active = False if self.her_goal_selection_strategy == "noher" else True
        self.her_n_sampled_goal = config['buffer']['her']['n_sampled_goal']
    
    def get_new_goals(self, episode, ep_t):
        if self.her_goal_selection_strategy == 'final':
            new_goals = []
            _, _, _, o2, _ = episode[-1]
            for _ in range(self.her_n_sampled_goal):
                new_goals.append(self.env.get_goal_state_from_obs(o2))
            return new_goals
        elif self.her_goal_selection_strategy == 'future' or self.her_goal_selection_strategy == 'future_once':
            new_goals = []
            for _ in range(self.her_n_sampled_goal):
                rand_future_transition = random.randint(ep_t, len(episode)-1)
                _, _, _, o2, _ = episode[rand_future_transition]
                new_goals.append(self.env.get_goal_state_from_obs(o2))
            return new_goals
        elif self.her_goal_selection_strategy == 'near':
            new_goals = []
            for _ in range(self.her_n_sampled_goal):
                rand_future_transition = random.randint(ep_t, min(len(episode)-1,ep_t+5))
                _, _, _, o2, _ = episode[rand_future_transition]
                new_goals.append(self.env.get_goal_state_from_obs(o2))
            return new_goals
        elif self.her_goal_selection_strategy == 'next':
            new_goals = []
            for _ in range(self.her_n_sampled_goal):
                _, _, _, o2, _ = episode[ep_t]
                new_goals.append(self.env.get_goal_state_from_obs(o2))
            return new_goals
    
    def add_virtial_experience(self,episode):

        if self.her_goal_selection_strategy == 'future_once':
            new_goals = self.get_new_goals(episode,0)
            for (o, a, r, o2, d) in episode:                  
                for new_goal in new_goals:
                    o_new = self.env.change_goal_in_obs(o, new_goal)
                    o2_new = self.env.change_goal_in_obs(o2, new_goal)
                    r_new, d_new = self.env.her_get_reward_and_done(o2_new) 
                    self.replay_buffer.store(o_new, a, r_new, o2_new, d_new)
        else:
            ep_t = 0
            for (o, a, r, o2, d) in episode:
                new_goals = self.get_new_goals(episode,ep_t)
                for new_goal in new_goals:
                    o_new = self.env.change_goal_in_obs(o, new_goal)
                    o2_new = self.env.change_goal_in_obs(o2, new_goal)
                    r_new, d_new = self.env.her_get_reward_and_done(o2_new) 
                    self.replay_buffer.store(o_new, a, r_new, o2_new, d_new)
                ep_t += 1