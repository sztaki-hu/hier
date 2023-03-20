import os
from os.path import dirname, abspath
current_dir = dirname(abspath(__file__))

import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--logdir", default="logs/0320_A_stack_blocks_sac/" ,help="Path of the config file")
parser.add_argument("--trainid", type=int, default=0 ,help="Train ID")
args = parser.parse_args()

path = os.path.join(current_dir, args.logdir,str(args.trainid),'csv','data.csv')
print(path)
df = pd.read_csv(path)

print(df.to_string()) 