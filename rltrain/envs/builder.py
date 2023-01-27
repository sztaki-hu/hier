ENVS = ['rlbench', 'simsim']

def make_env(config):

    env_name = config['environment']['name']

    assert env_name in ENVS 

    if env_name == 'rlbench':
        from rltrain.envs.RLBenchEnv import RLBenchEnv
        return RLBenchEnv(config)
    elif env_name == 'simsim':
        from rltrain.envs.SimSimEnv import SimSimEnv
        return SimSimEnv(config)