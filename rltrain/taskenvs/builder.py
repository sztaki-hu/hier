from typing import Dict, Union, Optional


import gymnasium as gym
from rltrain.taskenvs.Gym import Gym
from rltrain.taskenvs.GymPanda import GymPanda


def make_taskenv(config: Dict, config_framework: Dict) -> Union[Gym, GymPanda]:

    env_name = config['environment']['name']

    #assert env_name in config_framework['env_list'] 

    if env_name == 'gym':    
        return Gym(config, config_framework)
    elif env_name == 'gympanda':
        return GymPanda(config, config_framework)
    else:
        raise ValueError("[TaskEnv]: env_name: '" + str(env_name) + "' must be in : " + str(config_framework['env_list']))



    
    