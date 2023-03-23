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
parser.add_argument("--plotid", default="PLOT_0323_A" ,help="Id of plot")
parser.add_argument("--logdir", default="csv_to_plot" ,help="Path of the data folder")
parser.add_argument("--outdir", default="plots" ,help="Path of the output folder")
parser.add_argument("--taskname", default="stack_blocks_sac" ,help="Task name")
args = parser.parse_args()

create_folder(os.path.join(current_dir, args.outdir))

exps = []
exps.append({"name": ["0321_B"], "seed_num":[3], "color": "blue", "plotname": "baseline" })
exps.append({"name": ["0322_A","0322_B"], "seed_num":[3,3], "color": "orange", "plotname": "diff alpha"})
         
print(exps)

exp_color_list = []
for i in range(len(exps)):
    exp_color_list.append(exps[i]['color'])

graph_name = "test_glob_checkpoint_test_return"

test_ret = pd.DataFrame({'Step' : [], 'ExpName' : [],'Seed' : [],'Value' : []})

for i in range(len(exps)):
    running_id = 0
    for j in range(len(exps[i]['seed_num'])):
        for k in range(exps[i]['seed_num'][j]):
            graph_csv_name = "run-" + exps[i]['name'][j] + "_" + args.taskname +  "_" + str(k) + "_runs-tag-" + graph_name + ".csv"
            path = os.path.join(current_dir, args.logdir,graph_csv_name)
            pivot = pd.read_csv(path)
            pivot = pivot.drop(columns=['Wall time'])     

            pivot['ExpName'] = exps[i]['plotname']     
            pivot['Seed'] = str(running_id)
            running_id += 1
            pivot['Step'] /= 1000

            test_ret = test_ret.append(pivot, ignore_index = True)

print(test_ret.to_string())
print(test_ret.head())

# Separate plotting ########################## 

fig, _ = plt.subplots(figsize=(14,6))

for i in range(len(exps)):
    running_id = 0
    for j in range(len(exps[i]['seed_num'])):
        for k in range(exps[i]['seed_num'][j]):
            pivot=test_ret[test_ret["ExpName"] == exps[i]['plotname']]  
            pivot=pivot[pivot["Seed"] == str(running_id)] 
            
            if running_id == 0:
                plt.plot(pivot['Step'],pivot['Value'],color=exps[i]['color'],label=exps[i]['plotname'])
            else:
                plt.plot(pivot['Step'],pivot['Value'],color=exps[i]['color'])
            
            running_id += 1

plt.legend(title='Labels', bbox_to_anchor=(1, 1.01), loc='upper left')
plt.xlabel("Step (x1000)")
plt.ylabel("Average test return")
plt.title("Test returns")
figname = args.plotid + '_test_res_separate.png'
plt.savefig(os.path.join(current_dir, args.outdir, figname))
plt.show()

# SD plot ###################################

fig, _ = plt.subplots(figsize=(14,6))
sns.lineplot(data=test_ret, x="Step", y="Value", hue="ExpName", errorbar=('ci', 95), palette=exp_color_list)

plt.legend(title='Labels', bbox_to_anchor=(1, 1.01), loc='upper left')
plt.xlabel("Step (x1000)")
plt.ylabel("Average test return")
plt.title("Test returns with sd 95")
figname = args.plotid + '_test_res_std.png'
plt.savefig(os.path.join(current_dir, args.outdir, figname))
plt.show()

