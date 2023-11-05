import os
import numpy as np
import gymnasium as gym
import panda_gym
import argparse

from stable_baselines3 import HerReplayBuffer
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
#from stable_baselines3.common.noise import NormalActionNoise

from rltrain.utils.utils import init_cuda, save_yaml, load_yaml


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="cfg_exp/auto/config.yaml" ,help="Path of the config file")
    parser.add_argument("--explist", default="cfg_exp/auto/exp_list.yaml" ,help="Path of the config file")
    parser.add_argument("--trainid", type=int, default=0 ,help="Train ID")
    args = parser.parse_args()

    # Get experiments
    current_dir = os.path.dirname(os.path.abspath(__file__))
    exp_list = load_yaml(os.path.join(current_dir,args.explist))

    agents = exp_list['agents']
    envs = list(exp_list['tasks'].keys())
    her_strategies = exp_list['her_strategies']

    for env_name in envs:
        for task_name in exp_list['tasks'][env_name]:
            for agent_type in agents:
                for her_strategy in her_strategies:
                    # Load config
                    config_path = os.path.join(current_dir,args.config)
                    config = load_yaml(config_path)

                    config['environment']['name'] = env_name
                    config['environment']['task']['name'] = task_name
                    config['agent']['type'] = agent_type
                    config['buffer']['her']['goal_selection_strategy'] = her_strategy

                    print(config)

                    # Create log folder
                    logname = '_'.join(('SB3',
                                        config['general']['exp_name'],
                                        config['environment']['task']['name'],
                                        config['agent']['type'],
                                        config['buffer']['her']['goal_selection_strategy']))

                    # Experiment folder
                    logdir = config['general']['logdir']
                    os.makedirs(os.path.join(current_dir,logdir,logname,str(args.trainid)), exist_ok=True)

                    # Save config
                    save_yaml(os.path.join(current_dir, logdir,logname,"config.yaml"),config)
                    
                    # Backup model folder
                    eval_log_dir = os.path.join(current_dir,logdir, logname,str(args.trainid),"model_backup")
                    os.makedirs(eval_log_dir, exist_ok=True)

                    # Tensorboard
                    tb_log_dir = os.path.join(current_dir,logdir,logname,str(args.trainid),"runs")
                    os.makedirs(tb_log_dir, exist_ok=True)

                    # Init cuda
                    init_cuda(config["hardware"]["gpu"][args.trainid],config["hardware"]["cpu_min"][args.trainid],config["hardware"]["cpu_max"][args.trainid])

                    # Init Env
                    env_id = config["environment"]["task"]["name"]
                    n_training_envs = 1
                    n_eval_envs = 1
                    seed = config["general"]["seed"]

                    # Initialize a vectorized training environment with default parameters
                    train_env = make_vec_env(env_id, n_envs=n_training_envs, seed=seed)

                    # Separate evaluation env, with different parameters passed via env_kwargs
                    # Eval environments can be vectorized to speed up evaluation.
                    eval_env = make_vec_env(env_id, n_envs=n_eval_envs, seed=seed)

                    # Create callback that evaluates agent for 5 episodes every 500 training environment steps.
                    # When using multiple training environments, agent will be evaluated every
                    # eval_freq calls to train_env.step(), thus it will be evaluated every
                    # (eval_freq * n_envs) training steps. See EvalCallback doc for more information.
                    eval_callback = EvalCallback(eval_env, best_model_save_path=eval_log_dir,
                                                log_path=eval_log_dir, eval_freq=max(int(float(config["eval"]["freq"])) // n_training_envs, 1),
                                                n_eval_episodes=config["eval"]["num_episodes"], deterministic=True,
                                                render=False)

                    # Init HER
                    if type(train_env.observation_space) is not gym.spaces.dict.Dict: config["buffer"]["her"]["goal_selection_strategy"] = "noher"

                    replay_buffer_class = HerReplayBuffer if config["buffer"]["her"]["goal_selection_strategy"] != "noher" else None
                    replay_buffer_kwargs = replay_buffer_kwargs=dict(
                            n_sampled_goal=config["buffer"]["her"]["n_sampled_goal"],
                            goal_selection_strategy=config["buffer"]["her"]["goal_selection_strategy"],
                            ) if config["buffer"]["her"]["goal_selection_strategy"] != "noher" else None

                    # Init Obs type
                    policy_type = "MultiInputPolicy" if type(train_env.observation_space) is gym.spaces.dict.Dict else "MlpPolicy"

                    # Act space
                    act_dim = train_env.action_space.shape[0]

                    # Init Agent
                    if config["agent"]["type"] == "sac":
                        from stable_baselines3 import SAC
                        model = SAC(
                        policy_type,
                        train_env,
                        replay_buffer_class=replay_buffer_class,
                        replay_buffer_kwargs=replay_buffer_kwargs,
                        verbose=1,
                        tensorboard_log=tb_log_dir,
                        buffer_size=int(float(config["buffer"]["replay_buffer_size"])),
                        batch_size=config["trainer"]["batch_size"],
                        learning_starts=int(float(config["trainer"]["update_after"])),
                        train_freq=(int(float(config["trainer"]["update_every"])),"step"),
                        stats_window_size=config["logger"]["rollout"]["stats_window_size"],
                        learning_rate=float(config["agent"]["learning_rate"]),
                        gamma=config["agent"]["gamma"],
                        tau = 1.0 - config["agent"]["polyak"], 
                        ent_coef=config["agent"]["sac"]["alpha"],    
                        policy_kwargs=dict(net_arch=config["agent"]["hidden_sizes"]),
                        )
                        
                    elif config["agent"]["type"] == "td3":
                        from stable_baselines3 import TD3
                        from stable_baselines3.common.noise import NormalActionNoise, VectorizedActionNoise
                        model = TD3(
                        policy_type,
                        train_env,
                        replay_buffer_class=replay_buffer_class,
                        replay_buffer_kwargs=replay_buffer_kwargs,
                        verbose=1,
                        tensorboard_log=tb_log_dir,
                        buffer_size=int(float(config["buffer"]["replay_buffer_size"])),
                        batch_size=config["trainer"]["batch_size"],
                        learning_starts=int(float(config["trainer"]["update_after"])),
                        train_freq=(int(float(config["trainer"]["update_every"])),"step"),
                        stats_window_size=config["logger"]["rollout"]["stats_window_size"],
                        learning_rate=float(config["agent"]["learning_rate"]),
                        tau = 1.0 - config["agent"]["polyak"], 
                        gamma=config["agent"]["gamma"],
                        action_noise = VectorizedActionNoise(NormalActionNoise(np.zeros(act_dim),np.ones(act_dim)*config["agent"]["td3"]["act_noise"]),n_training_envs),
                        policy_delay = config["agent"]["td3"]["policy_delay"],
                        target_policy_noise = config["agent"]["td3"]["target_noise"],
                        target_noise_clip = config["agent"]["td3"]["noise_clip"],
                        policy_kwargs=dict(net_arch=config["agent"]["hidden_sizes"]),
                        )


                    # Save config
                    save_yaml(os.path.join(current_dir, logdir,logname,"config.yaml"),config)

                    # Train
                    model.learn(int(float(config["trainer"]["total_timesteps"])), progress_bar=True, callback=eval_callback)

                    # Save last model
                    model.save(os.path.join(eval_log_dir,logname+"_last_model"))

if __name__ == '__main__':
    main()
