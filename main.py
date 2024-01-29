import os
import argparse
import torch
import yaml
import datetime
from typing import Dict

from rltrain.utils.utils import init_cuda, print_torch_info, wait_for_datetime
from rltrain.logger.logger import Logger
from rltrain.runners.sampler_trainer_tester import SamplerTrainerTester

def load_yaml(file: str) -> Dict:
    if file is not None:
        with open(file) as f:
            return yaml.load(f, Loader=yaml.UnsafeLoader)
    return {}

def main(args: argparse.Namespace) -> int:

    current_dir = os.path.dirname(os.path.abspath(__file__))

    exp = load_yaml(args.exppath) if args.exppath != 'None' else None

    for _ in range(args.seednum):

        # Init logger ###############################################x
        logger = Logger(current_dir = current_dir, configpath = args.config, display_mode = False, exp = exp, is_test_config = False)

        config = logger.get_config()

        config_framework = logger.get_config_framework()

        # Init CUDA and torch and np ##################################
        init_cuda(config['hardware']['gpu'][args.hwid],config['hardware']['cpu_min'][args.hwid],config['hardware']['cpu_max'][args.hwid])

        print_torch_info(logger)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.print_logfile(str(device)) 

        samplerTrainerTester = SamplerTrainerTester(device,logger,config,config_framework,)

        samplerTrainerTester.start()
    
        logger.print_logfile(message = "##################################", level = "info", terminal = False) 
        logger.print_logfile(message = "Experiment finished!", level = "info", terminal = False) 
    return 1

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="cfg_exp/single/config.yaml", help="Path of the config file")
    parser.add_argument("--hwid", type=int, default=0, help="The id of the GPU")
    parser.add_argument("--seednum", type=int, default=1, help="The number of random seeds")
    parser.add_argument("--exppath", type=str, default='None', help="exppath")
    parser.add_argument("--delayed", type=bool, default=False, help="Time delay to start the training")
    args = parser.parse_args()

    if args.delayed == False:
        main(args)
    else:

        start_datetime = datetime.datetime(2024, 1, 29, 17, 37, 20)
        wait_for_datetime(start_datetime)

        main(args)