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
parser.add_argument("--plotid", default="test01" ,help="Id of plot")
parser.add_argument("--title", default="Effect of different alpha values" ,help="Title of the plots")
parser.add_argument("--logdir", default="csv_to_plot" ,help="Path of the data folder")
parser.add_argument("--outdir", default="plots" ,help="Path of the output folder")
args = parser.parse_args()

create_folder(os.path.join(current_dir, args.outdir))

exps = []

# ALPHA
exps.append({"name": ["0803_A_auto_PandaReach-v3_sac"], "seed_num":[3], "color": ["blue"], "plotname": "PandaReach-v3 sac"})
exps.append({"name": ["0803_A_auto_PandaReach-v3_td3"], "seed_num":[3], "color": ["cornflowerblue"], "plotname":"PandaReach-v3 td3"})
exps.append({"name": ["0803_A_auto_PandaReachDense-v3_sac"], "seed_num":[3], "color": ["green"], "plotname": "PandaReachDense-v3 sac"})
exps.append({"name": ["0803_A_auto_PandaReachDense-v3_td3"], "seed_num":[3], "color": ["limegreen"], "plotname": "PandaReachDense-v3 td3"})
exps.append({"name": ["0803_A_auto_PandaReachJoints-v3_sac"], "seed_num":[3], "color": ["magenta"], "plotname": "PandaReachJoints-v3 sac"})
exps.append({"name": ["0803_A_auto_PandaReachJoints-v3_td3"], "seed_num":[3], "color": ["orchid"], "plotname": "PandaReachJoints-v3 td3"})
exps.append({"name": ["0803_A_auto_PandaReachJointsDense-v3_sac"], "seed_num":[3], "color": ["red"], "plotname": "PandaReachJointsDense-v3 sac"})
exps.append({"name": ["0803_A_auto_PandaReachJointsDense-v3_td3"], "seed_num":[3], "color": ["maroon"], "plotname": "PandaReachJointsDense-v3 td3"})

exp_test_color_list = []
for i in range(len(exps)):
    exp_test_color_list.append(exps[i]['color'][0])


graph_name = "test_success_rate"

test_ret = pd.DataFrame({'Step' : [], 'ExpName' : [],'Seed' : [],'Value' : []})

for i in range(len(exps)):
    running_id = 0
    for j in range(len(exps[i]['seed_num'])):
        for k in range(exps[i]['seed_num'][j]):
            graph_csv_name = "run-" + exps[i]['name'][j] +  "_" + str(k) + "_runs-tag-" + graph_name + ".csv"
            path = os.path.join(current_dir, args.logdir,graph_csv_name)
            pivot = pd.read_csv(path)
            pivot = pivot.drop(columns=['Wall time'])     

            pivot['ExpName'] = exps[i]['plotname']     
            pivot['Seed'] = str(running_id)
            running_id += 1
            #pivot['Step'] /= 1000

            test_ret = test_ret.append(pivot, ignore_index = True)

print(test_ret.to_string())
print(test_ret.head())


# Separate plotting ########################## 

fig, _ = plt.subplots(figsize=(10,8))

for i in range(len(exps)):
    running_id = 0
    for j in range(len(exps[i]['seed_num'])):
        for k in range(exps[i]['seed_num'][j]):
            pivot=test_ret[test_ret["ExpName"] == exps[i]['plotname']]  
            pivot=pivot[pivot["Seed"] == str(running_id)]
            
            if running_id == 0:
                plt.plot(pivot['Step'],pivot['Value'],color=exps[i]['color'][0],label=exps[i]['plotname'])
            else:
                plt.plot(pivot['Step'],pivot['Value'],color=exps[i]['color'][0])
            
            running_id += 1

plt.legend(title='Labels', bbox_to_anchor=(1, 1.01), loc='upper left')
plt.xlabel("Step (1 step is 5000 t)")
plt.ylabel("Average test return")
plt.title(args.title)
figname = args.plotid + '_test_res_separate.png'
plt.savefig(os.path.join(current_dir, args.outdir, figname), bbox_inches='tight')
plt.show()

# SD plot ###################################

fig, _ = plt.subplots(figsize=(10,8))

sns.lineplot(data=test_ret, x="Step", y="Value", hue="ExpName", errorbar=('ci', 95), palette=exp_test_color_list)

plt.legend(title='Labels', bbox_to_anchor=(1, 1.01), loc='upper left')
plt.xlabel("Step (1 step is 5000 t)")
plt.ylabel("Average test return")
plt.title(args.title + " with sd")
figname = args.plotid + '_test_res_std.png'
plt.savefig(os.path.join(current_dir, args.outdir, figname),bbox_inches='tight')
plt.show()

