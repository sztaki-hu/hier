import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--configlistdir", default="cfg/lists/A/" ,help="Path of the config file dir")
parser.add_argument("--trainid", default="0" ,help="Trainid")
args = parser.parse_args()

for i in range(3):
    #print("python3 main.py --configfile " + args.configlistdir + str(i+1) + ".yaml --trainid " + str(args.trainid))
    os.system("python3 main.py --configfile " + args.configlistdir + str(i+1) + ".yaml --trainid " + str(args.trainid))

