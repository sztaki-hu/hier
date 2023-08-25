from os.path import dirname, abspath
import os
import gymnasium as gym
import panda_gym
import numpy as np
import time
from rltrain.envs.GymPanda import GymPanda
from rltrain.utils.utils import load_yaml


current_dir = dirname(abspath(__file__))

config = load_yaml(os.path.join(current_dir,"cfg_exp/manual/config.yaml"))
config['sampler']['max_ep_len'] = 50
config['environment']['headless'] = False
config_framework = load_yaml(os.path.join(current_dir,'cfg_framework','config_framework.yaml'))

env = GymPanda(config, config_framework)
observation  = env.reset()

robot_joints,desired_goal,object_position = env.save_state()

for i in range(200):

    #action = env.action_space.sample() # random action
    if i == 10: 
        robot_joints,desired_goal,object_position = env.save_state()
    
    if i >= 10:  
        action = np.array([0.1,0.0,0.0])
    else:
        action = np.array([0.0,0.1,0.0])
    

    observation, reward, terminated, truncated, info = env.step(action)

    

    if terminated or truncated:
        observation = env.reset()
        env.restore_state(robot_joints,desired_goal,object_position)

env.shuttdown()