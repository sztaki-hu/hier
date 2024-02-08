import os
from os.path import dirname, abspath

import numpy as np
import torch
import argparse
import rltrain.agents.sac.core as core
import pandas as pd

current_dir = dirname(abspath(__file__))
current_dir = os.path.abspath(os.path.join(current_dir, os.pardir))

from rltrain.utils.utils import init_cuda, print_torch_info, get_best_seed
from rltrain.logger.logger import Logger
from rltrain.utils.eval import Eval

from rltrain.agents.builder import make_agent

import datetime
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

import torch

def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(path + ' folder is created!')
    else:
        print(path + ' folder already exists!')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    
    #parser.add_argument("--config", default="logs/"+"_".join(['XXX_0208','sac','noher','nohier','fix','max','7x7_base','noper','sparse','PandaPickAndPlace-v3']),help="Path of the config file")

    parser.add_argument("--config", default="logs/"+"_".join(['0207_A','sac','final','predefined','fix','max','noper','sparse','FetchPickAndPlace-v2']),help="Path of the config file")
    #parser.add_argument("--config", default="logs/"+"_".join(['0207_B','sac','final','predefined','fix','max','7x7_wall','noper','sparse','PointMaze_UMaze-v3']),help="Path of the config file")


    parser.add_argument("--figid", default="1106_A" ,help="Fig id")
    parser.add_argument("--hwid", type=int, default=0 ,help="Hardware id")
    parser.add_argument("--seedid", type=int, default=0 ,help="seedid")
    parser.add_argument("--bestfromseeds", type=bool, default=True ,help="best of seeds flag")
    parser.add_argument("--outdir", default="results/output/eval" ,help="Path of the output folder")
    # Example: python3 main.py --configfile /cfg/alma.yaml 0
    args = parser.parse_args()

    if args.bestfromseeds:
        seedid = get_best_seed(current_dir, args.config)
    else:
        seedid = args.seedid

    create_folder(os.path.join(current_dir, args.outdir))
    create_folder(os.path.join(current_dir, args.outdir, args.figid))

    logger = Logger(current_dir = current_dir, configpath = args.config, display_mode = True, seed = seedid)

    config = logger.get_config()
    config_framework = logger.get_config_framework()

    # Init CUDA
    init_cuda(config['hardware']['gpu'][args.hwid],config['hardware']['cpu_min'][args.hwid],config['hardware']['cpu_max'][args.hwid])

    print_torch_info(logger, display_mode = True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    torch.set_num_threads(torch.get_num_threads())

    # Init Agent
    agent = make_agent(device,config, config_framework)

    # Init Trainer
    tester = Eval(device, logger, config, config_framework)

    # Test Agent
    tester.eval_agent(model_name="best_model",num_display_episode=10, headless=False, time_delay = 0.0, current_dir = current_dir, outdir = args.outdir, figid = args.figid)

    # Test Agent with Stats (only gympanda)
    #tester.eval_agent_stats(model_name="best_model",num_display_episode=10, headless=True, time_delay = 0.05, current_dir = current_dir, outdir = args.outdir, figid = args.figid)



    
