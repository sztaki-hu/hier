from typing import Dict, Union, Optional


import gymnasium as gym
from rltrain.taskenvs.GymMaze import GymMaze
from rltrain.taskenvs.GymPanda import GymPanda
from rltrain.taskenvs.GymFetch import GymFetch


def make_taskenv(config: Dict, config_framework: Dict) -> Union[GymMaze, GymPanda, GymFetch]:

    env_name = config['environment']['name']

    #assert env_name in config_framework['env_list'] 

    if env_name == 'gymmaze':    
        return GymMaze(config, config_framework)
    elif env_name == 'gympanda':
        return GymPanda(config, config_framework)
    elif env_name == 'gymfetch':
        return GymFetch(config, config_framework)
    else:
        raise ValueError("[TaskEnv]: env_name: '" + str(env_name) + "' must be in : " + str(config_framework['env_list']))



    
    