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
parser.add_argument("--plotid", default="PLOT_0322_B" ,help="Id of plot")
parser.add_argument("--logdir", default="csv_to_plot" ,help="Path of the data folder")
parser.add_argument("--outdir", default="plots" ,help="Path of the output folder")
parser.add_argument("--taskname", default="stack_blocks_sac" ,help="Task name")
args = parser.parse_args()

create_folder(os.path.join(current_dir, args.outdir))

# expname_list = ["0320_A","0320_B","0320_C","0320_D","0320_E","0320_F"]
# expcolor_list = ["blue","orange","magenta","green","red","yellow"]
# seeds_list = [2,2,2,2,2,3]

# expname_list = ["0320_C","0320_F","0321_A","0321_B"]
# expcolor_list = ["blue","orange","magenta","green","red","yellow"]
# seeds_list = [2,3,3,3]

expname_list = ["0320_B","0320_C"]
expcolor_list = ["blue","orange","magenta","green","red","yellow"]
seeds_list = [2,2]

graph_name = "test_glob_checkpoint_test_return"

test_ret = pd.DataFrame({'Step' : [], 'ExpName' : [],'Seed' : [],'Value' : []})
for expname,seed_num in zip(expname_list,seeds_list):
    for i in range(seed_num):
        graph_csv_name = "run-" + expname + "_" + args.taskname +  "_" + str(i) + "_runs-tag-" + graph_name + ".csv"
        path = os.path.join(current_dir, args.logdir,graph_csv_name)
        pivot = pd.read_csv(path)
        pivot = pivot.drop(columns=['Wall time'])
        pivot['ExpName'] = expname
        pivot['Seed'] = str(i)
        pivot['Step'] /= 1000

        test_ret = test_ret.append(pivot, ignore_index = True)

#print(test_ret.to_string())
#print(test_ret.head())
   

# Separate plotting ########################## 

fig, _ = plt.subplots(figsize=(14,6))
for i in range(len(expname_list)):
    for j in range(seeds_list[i]):

        pivot=test_ret[test_ret["ExpName"] == expname_list[i]]    
        pivot=pivot[pivot["Seed"] == str(j)] 
        if j == 0:
            plt.plot(pivot['Step'],pivot['Value'],color=expcolor_list[i],label=expname_list[i])
        else:
            plt.plot(pivot['Step'],pivot['Value'],color=expcolor_list[i])

plt.legend(title='Labels', bbox_to_anchor=(1, 1.01), loc='upper left')
plt.xlabel("Step (x1000)")
plt.ylabel("Average test return")
plt.title("Test returns")
figname = args.plotid + '_test_res_separate.png'
plt.savefig(os.path.join(current_dir, args.outdir, figname))
plt.show()

# SD plot ###################################

fig, _ = plt.subplots(figsize=(14,6))
sns.lineplot(data=test_ret, x="Step", y="Value", hue="ExpName", errorbar=('ci', 95), palette=expcolor_list)

plt.legend(title='Labels', bbox_to_anchor=(1, 1.01), loc='upper left')
plt.xlabel("Step (x1000)")
plt.ylabel("Average test return")
plt.title("Test returns with sd 95")
figname = args.plotid + '_test_res_std.png'
plt.savefig(os.path.join(current_dir, args.outdir, figname))
plt.show()

