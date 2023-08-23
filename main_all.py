import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="cfg_exp/manual/config.yaml" ,help="Path of the config file")
parser.add_argument("--trainid", type=int, default=0 ,help="Train ID")
args = parser.parse_args()

# Own Framework
os.system("python3 main_auto.py --config " + args.config + " --trainid " + str(args.trainid))
#print("python3 main_auto.py --config " + args.config + " --trainid " + str(args.trainid))

# SB3
os.system("python3 main_sb_auto.py --config " + args.config + " --trainid " + str(args.trainid))
#print("python3 main_sb_auto.py --config " + args.config + " --trainid " + str(args.trainid))
