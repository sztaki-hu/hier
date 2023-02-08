from os.path import dirname, abspath

import numpy as np
import torch
import argparse

from multiprocessing import Process
from rltrain.buffers.demo import Demo

from rltrain.utils.utils import init_cuda, print_torch_info
from rltrain.logger.logger import Logger

import datetime
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

import torch
    

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--configfile", default="/cfg/config.yaml" ,help="Path of the config file")
    parser.add_argument("--trainid", type=int, default=0 ,help="Train ID")
    # Example: python3 main.py --configfile /cfg/alma.yaml
    args = parser.parse_args()

    # Init logger ###############################################x
    current_dir = dirname(abspath(__file__))
    config_path = current_dir + args.configfile
    #config_path = current_dir + "/cfg/config_test.yaml"
    logger = Logger(current_dir = current_dir, config_path = config_path, trainid = args.trainid, light_mode = True)
    config = logger.get_config()

    #print(config['environment']['task']['params'])
    
    # Init CUDA and torch and np ##################################
    init_cuda(config['hardware']['gpu'][args.trainid],config['hardware']['cpu_min'][args.trainid],config['hardware']['cpu_max'][args.trainid])

    print_torch_info()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    torch.set_num_threads(torch.get_num_threads())

    torch.manual_seed(config['general']['seed'])
    np.random.seed(config['general']['seed'])

    # Init Demo Buffer ################################################
    if config['demo']['demo_use']:
        demo = Demo(logger,config)
        if demo.demo_create:
            demo.clean_up_old_demo()
            demo.create_demos()
            # demo_p = Process(target=demo.create_demos, args=[])
            # demo_p.start()
            # demo_p.join()  
        else:
            demoBuffer = demo.load_demos()
            batch = demoBuffer.sample_batch(batch_size=10)
            print(batch)
                     

if __name__ == '__main__':
    main()

    
