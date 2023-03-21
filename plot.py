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
parser.add_argument("--taskname", default="stack_blocks_sac" ,help="Task name")
args = parser.parse_args()

create_folder(os.path.join(current_dir, args.outdir))

expname_list = ["0320_A","0320_F"]
expcolor_list = ["blue","orange"]
seeds_list = [2,3]

graph_name = "test_glob_checkpoint_test_return"
test_ret_list = []
for expname,seed in zip(expname_list,seeds_list):
    print(expname)
    print(seed)
    graph_csv_name = "run-" + expname + "_" + args.taskname +  "_" + str(0) + "_runs-tag-" + graph_name + ".csv"
    path = os.path.join(current_dir, args.logdir,graph_csv_name)
    test_ret = pd.read_csv(path)
    test_ret['Step'] /= 1000
    test_ret = test_ret.rename(columns={"Value": "Value_0"})
    test_ret = test_ret.drop(columns=['Wall time'])

    for i in range(1,seed):
        graph_csv_name = "run-" + expname + "_" + args.taskname +  "_" + str(i) + "_runs-tag-" + graph_name + ".csv"
        path = os.path.join(current_dir, args.logdir,graph_csv_name)
        pivot = pd.read_csv(path)
        test_ret['Value_'+str(i)] = pivot['Value']

    test_ret["id"] = test_ret.index
    print(test_ret.head())
    test_ret_list.append(test_ret)

# Separate plotting ########################## 

for i in range(len(test_ret_list)):
    for j in range(seeds_list[i]):
        #sns.lineplot(data=test_ret_list[i], x="Step", y="Value_"+str(j),palette=expcolor_list[i])
        plt.plot(test_ret_list[i]['Step'],test_ret_list[i]['Value_'+str(j)],color=expcolor_list[i])
plt.xlabel("Step (x1000)")
plt.ylabel("Average test return")
plt.title("Test returns")
figname = 'test_res_separate.png'
plt.savefig(os.path.join(current_dir, args.outdir, figname))
plt.show()
plt.clf()

# SD plot ###################################
test_ret_wide = []
for i in range(len(test_ret_list)):
    test_ret_wide.append(pd.wide_to_long(test_ret_list[i], stubnames='Value', i='id', j="Step", sep='_'))

for i in range(len(test_ret_wide)):
    sns.lineplot(data=test_ret_wide[i], x="Step", y="Value", errorbar=('ci', 95))
plt.xlabel("Step (x1000)")
plt.ylabel("Average test return")
plt.title("Test returns with sd 95")
figname = 'test_res_std.png'
plt.savefig(os.path.join(current_dir, args.outdir, figname))
plt.show()
plt.clf()
