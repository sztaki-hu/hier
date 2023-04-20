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
parser.add_argument("--plotid", default="PLOT_demos_new" ,help="Id of plot")
parser.add_argument("--logdir", default="csv_to_plot" ,help="Path of the data folder")
parser.add_argument("--outdir", default="plots" ,help="Path of the output folder")
parser.add_argument("--taskname", default="stack_blocks_sac" ,help="Task name")
args = parser.parse_args()

create_folder(os.path.join(current_dir, args.outdir))

exps = []
# AGENT NUM
# exps.append({"name": ["0323_B","0323_C"], "seed_num":[3,3], "color": "lime", "plotname": "1 agent"})
# exps.append({"name": ["0323_A"], "seed_num":[3], "color": "green", "plotname": "1 agent with fallback" })
# exps.append({"name": ["0321_B"], "seed_num":[3], "color": "blue", "plotname": "3 agent" })
# exps.append({"name": ["0322_A","0322_B"], "seed_num":[3,3], "color": "cyan", "plotname": "3 agent diff alpha"})
# exps.append({"name": ["0323_D"], "seed_num":[3], "color": "magenta", "plotname": "6 agent (replay ratio 0.5)"}) 

# LEARNING RATE
# exps.append({"name": ["0325_B"], "seed_num":[3], "color": "orange", "plotname": "1 agent; lr=0.0005" })
# exps.append({"name": ["0323_B","0323_C","0324_D"], "seed_num":[3,3,3], "color": "cyan", "plotname": "1 agent; lr=0.001"})
# exps.append({"name": ["0323_A","0324_C"], "seed_num":[3,3], "color": "blue", "plotname": "1 agent; fallback: True; lr=0.001" })
# exps.append({"name": ["0325_A"], "seed_num":[3], "color": "magenta", "plotname": "1 agent; lr=0.002" })
# exps.append({"name": ["0324_A"], "seed_num":[3], "color": "purple", "plotname": "1 agent; lr=0.005" })
# exps.append({"name": ["0324_B"], "seed_num":[3], "color": "indigo", "plotname": "1 agent with fallback: True; lr=0.005"})

# UPDATE FACTOR
# exps.append({"name": ["0325_C"], "seed_num":[3], "color": "orange", "plotname": "1 agent; uf = 0.5" })
# exps.append({"name": ["0323_B","0323_C","0324_D"], "seed_num":[3,3,3], "color": "cyan", "plotname": "1 agent; uf=1.0"})
# exps.append({"name": ["0323_A","0324_C"], "seed_num":[3,3], "color": "blue", "plotname": "1 agent; fallback: True; uf=1.0"})
# exps.append({"name": ["0325_D"], "seed_num":[3], "color": "lime", "plotname": "1 agent; uf = 2.0"})
# exps.append({"name": ["0326_A"], "seed_num":[3], "color": "green", "plotname": "1 agent; fallback: True; uf = 2.0"})
# exps.append({"name": ["0325_E"], "seed_num":[3], "color": "magenta", "plotname": "1 agent; uf = 4.0" })
# exps.append({"name": ["0325_F"], "seed_num":[3], "color": "purple", "plotname": "1 agent; uf = 6.0"})

# ALPHA
# exps.append({"name": ["0326_B"], "seed_num":[3], "color": "orange", "plotname": "1 agent fb: False; uf: 2.0; alpha: 0.1"})
# exps.append({"name": ["0325_D"], "seed_num":[3], "color": "lime", "plotname": "1 agent fb: False; uf: 2.0; alpha: 0.2"})
# exps.append({"name": ["0326_A"], "seed_num":[3], "color": "green", "plotname": "1 agent fb: True; uf: 2.0; alpha: 0.2"})
# exps.append({"name": ["0326_C"], "seed_num":[3], "color": "blue", "plotname": "1 agent fb: False; uf: 2.0; alpha: 0.3"})
# exps.append({"name": ["0326_D"], "seed_num":[3], "color": "magenta", "plotname": "1 agent fb: False; uf: 2.0; alpha: 0.4"})

# DEMO
exps.append({"name": ["0325_D"], "seed_num":[3], "color": "orange", "plotname": "1 agent; 0325_D"})
exps.append({"name": ["0326_E_01","0326_E_02"], "seed_num":[3,3], "color": "green", "plotname": "1 agent; 0326_E"})
exps.append({"name": ["0326_F_01","0326_F_02"], "seed_num":[3,3], "color": "blue", "plotname": "1 agent; 0326_F"})
exps.append({"name": ["0326_G_01","0326_G_02"], "seed_num":[3,3], "color": "magenta", "plotname": "1 agent; 0326_G"})


         
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

fig, _ = plt.subplots(figsize=(16,6))

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
plt.savefig(os.path.join(current_dir, args.outdir, figname), bbox_inches='tight')
plt.show()

# SD plot ###################################

fig, _ = plt.subplots(figsize=(14,6))
sns.lineplot(data=test_ret, x="Step", y="Value", hue="ExpName", errorbar=('ci', 50), palette=exp_color_list)

plt.legend(title='Labels', bbox_to_anchor=(1, 1.01), loc='upper left')
plt.xlabel("Step (x1000)")
plt.ylabel("Average test return")
plt.title("Test returns with sd")
figname = args.plotid + '_test_res_std.png'
plt.savefig(os.path.join(current_dir, args.outdir, figname),bbox_inches='tight')
plt.show()

