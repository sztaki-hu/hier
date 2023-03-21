import os
from os.path import dirname, abspath
current_dir = dirname(abspath(__file__))

import pandas as pd
import argparse

# Import seaborn
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()

def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(path + ' folder is created!')
    else:
        print(path + ' folder already exists!')

parser = argparse.ArgumentParser()
parser.add_argument("--logdir", default="csv_to_plot" ,help="Path of the data folder")
parser.add_argument("--outdir", default="plots" ,help="Path of the output folder")
parser.add_argument("--expname", default="0320_F_stack_blocks_sac",help="Path of the config file")
parser.add_argument("--trainnum", type=int, default=3 ,help="Train ID")
args = parser.parse_args()

create_folder(os.path.join(current_dir, args.outdir))

graph_name = "test_glob_checkpoint_test_return"
graph_csv_name = "run-" + str(args.expname) +  "_" + str(0) + "_runs-tag-" + graph_name + ".csv"
path = os.path.join(current_dir, args.logdir,graph_csv_name)
test_ret = pd.read_csv(path)
test_ret['Step'] /= 1000
test_ret = test_ret.rename(columns={"Value": "Value_0"})
test_ret = test_ret.drop(columns=['Wall time'])

for i in range(1,args.trainnum):
    graph_csv_name = "run-" + str(args.expname) +  "_" + str(i) + "_runs-tag-" + graph_name + ".csv"
    path = os.path.join(current_dir, args.logdir,graph_csv_name)
    pivot = pd.read_csv(path)
    
    # Using 'Address' as the column name
    # and equating it to the list
    test_ret['Value_'+str(i)] = pivot['Value']

test_ret["id"] = test_ret.index
print(test_ret.head())

# Separate plotting ########################## 

for i in range(args.trainnum):
    sns.lineplot(data=test_ret, x="Step", y="Value_"+str(i))
plt.xlabel("Step (x1000)")
plt.ylabel("Average test return")
plt.title("Test returns of different seeds")
figname = args.expname + '_test_res_separate.png'
plt.savefig(os.path.join(current_dir, args.outdir, figname))
plt.show()
plt.clf()

# SD plot ###################################
test_ret_wide = pd.wide_to_long(test_ret, stubnames='Value', i='id', j="Step", sep='_')

sns.lineplot(data=test_ret_wide, x="Step", y="Value", errorbar=('ci', 95))
plt.xlabel("Step (x1000)")
plt.ylabel("Average test return")
plt.title("Test returns with sd 95")
figname = args.expname + '_test_res_std.png'
plt.savefig(os.path.join(current_dir, args.outdir, figname))
plt.show()
plt.clf()
