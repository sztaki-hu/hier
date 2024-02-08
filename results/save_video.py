import os
from os.path import dirname, abspath
import gymnasium as gym
from numpngw import write_apng  # pip install numpngw
from tqdm import tqdm

current_dir = dirname(abspath(__file__))
current_dir = os.path.abspath(os.path.join(current_dir, os.pardir))

import torch
import panda_gym
import argparse

from rltrain.utils.utils import init_cuda, print_torch_info, get_best_seed
from rltrain.logger.logger import Logger
from rltrain.agents.builder import make_agent
from rltrain.utils.eval import Eval

def create_folder(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)
        print(path + ' folder is created!')
    else:
        print(path + ' folder already exists!')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    # Slide
    #parser.add_argument("--config", default="logs/"+"_".join(['1108_B','sac','selfpaced','final','predefined','prioritized','proportional','sparse','PandaSlide-v3']),help="Path of the config file")
    #parser.add_argument("--config", default="logs/"+"_".join(['1108_A','sac','nocl','final','nohier','fix','proportional','sparse','PandaSlide-v3']),help="Path of the config file")
    
    # Pick-and-Place
    #parser.add_argument("--config", default="logs/"+"_".join(['1109_A','sac','selfpaced','final','predefined','prioritized','noper','sparse','PandaPickAndPlace-v3']),help="Path of the config file")
    #parser.add_argument("--config", default="logs/"+"_".join(['1109_A','sac','nocl','final','nohier','fix','noper','sparse','PandaPickAndPlace-v3']),help="Path of the config file")
    
    #parser.add_argument("--config", default="logs/"+"_".join(['0207_A','sac','noher','predefined','fix','max','noper','sparse','FetchSlide-v2']),help="Path of the config file")
    parser.add_argument("--config", default="logs/"+"_".join(['0207_B','sac','noher','predefined','fix','max','7x7_S','noper','sparse','PointMaze_UMaze-v3']),help="Path of the config file")
    #parser.add_argument("--config", default="logs/"+"_".join(['XXX_0208','sac','noher','nohier','fix','max','7x7_base','noper','sparse','PandaPickAndPlace-v3']),help="Path of the config file")


    parser.add_argument("--hwid", type=int, default=0 ,help="Hardware id")
    parser.add_argument("--seedid", type=int, default=0 ,help="seedid")
    parser.add_argument("--bestfromseeds", type=bool, default=True ,help="best of seeds flag")
    parser.add_argument("--outdir", default="results/output/vids" ,help="Path of the output folder")
    # Example: python3 main.py --configfile /cfg/alma.yaml 0
    args = parser.parse_args()

    if args.bestfromseeds:
        seedid = get_best_seed(current_dir, args.config)
    else:
        seedid = args.seedid

    create_folder(os.path.join(current_dir, args.outdir))

    logger = Logger(current_dir = current_dir, configpath = args.config, display_mode = True, seed = seedid)

    config = logger.get_config()
    config_framework = logger.get_config_framework()

    # Init CUDA
    init_cuda(config['hardware']['gpu'][args.hwid],config['hardware']['cpu_min'][args.hwid],config['hardware']['cpu_max'][args.hwid])

    print_torch_info(logger, display_mode = True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    torch.set_num_threads(torch.get_num_threads())

    # Init Trainer
    tester = Eval(device, logger,config,config_framework)

    # Test Agent
    tester.record(model_name="best_model",
                      num_display_episode=10, 
                      current_dir = current_dir, 
                      frame_freq = 1,
                      outdir = args.outdir, 
                      save_name = "0207_Maze_S_IMG",
                      save_images = True,
                      save_video = False)
    