def make_env(config, config_framework):

    env_name = config['environment']['name']

    assert env_name in config_framework['env_list'] 

    if env_name == 'gym':
        import gymnasium as gym
        from rltrain.envs.Gym import Gym
        return Gym(config, config_framework)
    if env_name == 'gym_mod':
        import gymnasium as gym
        from rltrain.envs.Gym import Gym
        return Gym(config, config_framework)
    elif env_name == 'gympanda':
        import gymnasium as gym
        from rltrain.envs.GymPanda import GymPanda
        return GymPanda(config, config_framework)
    elif env_name == 'rlbench_joint':
        from rltrain.envs.RLBenchJoint import RLBenchJoint
        return RLBenchJoint(config, config_framework)

    
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
        import gymnasium as gym
        return gym.make(task_name) if headless == True else gym.make(task_name, render_mode="human") 
    
    elif env_name == 'gympanda':
        assert task_name in config_framework['task_list']['gympanda']
        import gymnasium as gym
        return gym.make(task_name) if headless == True else gym.make(task_name, render_mode="human")

    elif env_name == 'gym_mod':
        assert task_name in config_framework['task_list']['gym_mod']
        import gymnasium as gym
        if task_name == "Hopper-v4":
            from rltrain.tasks.hopper_v4 import HopperEnv
            return HopperEnv() if headless == True else HopperEnv(render_mode="human")
    
    elif env_name == 'rlbench_joint':
        assert task_name in config_framework['task_list']['rlbench_joint']

        from rlbench.observation_config import ObservationConfig, CameraConfig
        from pyrep.const import RenderMode

        from rlbench.environment import Environment
        from rlbench.action_modes.action_mode import  MoveArmThenGripper
        from rlbench.action_modes.arm_action_modes import JointVelocity
        from rlbench.action_modes.gripper_action_modes import Discrete, Closed

        if config['environment']['camera'] == True:
                cam_config = CameraConfig(rgb=True, depth=True, point_cloud=True, mask=False,image_size=(256, 256),
                                        render_mode=RenderMode.OPENGL)
                obs_config = ObservationConfig()
                obs_config.set_all(False)
                obs_config.joint_positions = False
                obs_config.gripper_pose = True
                obs_config.right_shoulder_camera = cam_config
                obs_config.left_shoulder_camera = cam_config
                obs_config.wrist_camera = cam_config
                obs_config.front_camera = cam_config
                #obs_config.task_low_dim_state=True
            
        else:
            obs_config = ObservationConfig()
            obs_config.set_all(False)
            obs_config.joint_positions = False
            obs_config.gripper_pose = True
            #obs_config.task_low_dim_state=True

        arm_action_mode = JointVelocity()

        gripper_action_mode = Closed()

        act_mode = MoveArmThenGripper(arm_action_mode,gripper_action_mode)

        return Environment(action_mode = act_mode, obs_config= obs_config,headless = config['environment']['headless'], robot_setup = 'ur3baxter')

        

