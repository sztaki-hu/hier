import os
from os.path import dirname, abspath
import gymnasium as gym
from numpngw import write_apng  # pip install numpngw
from tqdm import tqdm

import torch
import panda_gym
import argparse

from rltrain.utils.utils import init_cuda, print_torch_info
from rltrain.logger.logger import Logger
from rltrain.agents.builder import make_agent
from rltrain.utils.eval import Eval

def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(path + ' folder is created!')
    else:
        print(path + ' folder already exists!')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="logs/0928_A_PandaPush-v3_sac_controldiscrete_const" ,help="Path of the config file")
    parser.add_argument("--hwid", type=int, default=0 ,help="Hardware id")
    parser.add_argument("--seedid", type=int, default=3 ,help="seedid")
    parser.add_argument("--outdir", default="vids" ,help="Path of the output folder")
    # Example: python3 main.py --configfile /cfg/alma.yaml 0
    args = parser.parse_args()

    env = gym.make("PandaPush-v3", render_mode="rgb_array")
    images = []

    # Init logger 
    current_dir = dirname(abspath(__file__))

    create_folder(os.path.join(current_dir, args.outdir))

    logger = Logger(current_dir = current_dir, main_args = args, display_mode = True)
    config = logger.get_config()
    config_framework = logger.get_config_framework()

    # Init CUDA
    init_cuda(config['hardware']['gpu'][args.hwid],config['hardware']['cpu_min'][args.hwid],config['hardware']['cpu_max'][args.hwid])

    print_torch_info(logger)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    torch.set_num_threads(torch.get_num_threads())

    # Init Agent
    agent = make_agent(device,config,config_framework)

    # Init Trainer
    tester = Eval(agent,logger,config,config_framework)

    # Test Agent
    tester.save_video(model_name="best_model",num_display_episode=20, current_dir = current_dir, outdir = args.outdir, save_name = "video2.png")
    