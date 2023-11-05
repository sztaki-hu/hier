from typing import Dict, Union
import torch

from rltrain.agents.sac.agent import Agent as SAC
from rltrain.agents.td3.agent import Agent as TD3
from rltrain.agents.ddpg.agent import Agent as DDPG

def make_agent(device: torch.device, config: Dict, config_framework: Dict) -> Union[SAC,TD3,DDPG]:

    agent_type = config['agent']['type']
    #assert agent_type in config_framework['agent_list'] 

    if agent_type == 'sac':
        return SAC(device,config)
    elif agent_type == 'td3': 
        return TD3(device,config)
    elif agent_type == 'ddpg': 
        return DDPG(device,config)
    else:
        raise ValueError("[Agent]: agent_type: '" + str(agent_type) + "' must be in : " + str(config_framework['agent_list']))
   