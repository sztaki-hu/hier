from typing import Dict, Union, Optional


import gymnasium as gym
from rltrain.taskenvs.Gym import Gym
from rltrain.taskenvs.GymPanda import GymPanda


def make_taskenv(config: Dict, config_framework: Dict) -> GymPanda:

    env_name = config['environment']['name']

    assert env_name in config_framework['env_list'] 

    # if env_name == 'gym':    
    #     return Gym(config, config_framework)
    if env_name == 'gympanda':
        return GymPanda(config, config_framework)
    else:
        assert False



    
    