ENVS = ['rlbench', 'simsim','simsimv2','gym','gympanda','rlbenchjoint']

def make_env(config):

    env_name = config['environment']['name']

    assert env_name in ENVS 

    if env_name == 'rlbench':
        from rltrain.envs.RLBenchEnv import RLBenchEnv
        return RLBenchEnv(config)
    elif env_name == 'simsim':
        from rltrain.envs.SimSimEnv import SimSimEnv
        return SimSimEnv(config)
    elif env_name == 'simsimv2':
        from rltrain.envs.SimSimEnvV2 import SimSimEnvV2
        return SimSimEnvV2(config)
    elif env_name == 'gym':
        from rltrain.envs.Gym import Gym
        return Gym(config)
    elif env_name == 'gympanda':
        from rltrain.envs.GymPanda import GymPanda
        return GymPanda(config)
    elif env_name == 'rlbenchjoint':
        from rltrain.envs.RLBenchEnvJoint import RLBenchEnvJoint
        return RLBenchEnvJoint(config)