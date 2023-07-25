AGENT_TYPES = ['sac', 'td3']

def make_agent(id,device,config):

    agent_type = config['agent']['type']

    assert agent_type in AGENT_TYPES 

    if agent_type == 'sac':
        from rltrain.agents.sac.agent_v0 import Agent
        return Agent(id,device,config)
    elif agent_type == 'td3':
        from rltrain.agents.td3.agent import Agent
        return Agent(id,device,config)
   