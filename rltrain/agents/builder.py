def make_agent(id,device,config,config_framework):

    agent_type = config['agent']['type']

    assert agent_type in config_framework['agent_list'] 

    if agent_type == 'sac':
        from rltrain.agents.sac.agent_v0 import Agent
        return Agent(id,device,config)
    elif agent_type == 'td3':
        from rltrain.agents.td3.agent import Agent
        return Agent(id,device,config)
   