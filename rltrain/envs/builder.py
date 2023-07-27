import gymnasium as gym

def make_env(config, config_framework):

    env_name = config['environment']['name']

    assert env_name in config_framework['env_list'] 

    if env_name == 'gym':
        from rltrain.envs.Gym import Gym
        return Gym(config, config_framework)
    if env_name == 'gym_mod':
        from rltrain.envs.Gym import Gym
        return Gym(config, config_framework)
    elif env_name == 'gympanda':
        from rltrain.envs.GymPanda import GymPanda
        return GymPanda(config, config_framework)
    # elif env_name == 'rlbench':
    #     from rltrain.envs.RLBenchEnv import RLBenchEnv
    #     return RLBenchEnv(config)
    # elif env_name == 'simsim':
    #     from rltrain.envs.SimSimEnv import SimSimEnv
    #     return SimSimEnv(config)
    # elif env_name == 'simsimv2':
    #     from rltrain.envs.SimSimEnvV2 import SimSimEnvV2
    #     return SimSimEnvV2(config)   
    # elif env_name == 'rlbenchjoint':
    #     from rltrain.envs.RLBenchEnvJoint import RLBenchEnvJoint
    #     return RLBenchEnvJoint(config)

def make_task(config,config_framework):
    
    env_name = config['environment']['name']
    task_name = config['environment']['task']['name']

    headless = config['environment']['headless']

    assert env_name in config_framework['env_list']
    
    if env_name == 'gym':
        assert task_name in config_framework['task_list']['gym']
        return gym.make(task_name) if headless == True else gym.make(task_name, render_mode="human") 
    
    elif env_name == 'gympanda':
        assert task_name in config_framework['task_list']['gympanda']
        return gym.make(task_name) if headless == True else gym.make(task_name, render_mode="human")

    elif env_name == 'gym_mod':
        assert task_name in config_framework['task_list']['gym_mod']
        if task_name == "Hopper-v4":
            from rltrain.tasks.hopper_v4 import HopperEnv
            return HopperEnv() if headless == True else HopperEnv(render_mode="human")

