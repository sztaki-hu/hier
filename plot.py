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
parser.add_argument("--plotid", default="alphav2_01" ,help="Id of plot")
parser.add_argument("--title", default="Effect of different alpha values" ,help="Title of the plots")
parser.add_argument("--logdir", default="csv_to_plot" ,help="Path of the data folder")
parser.add_argument("--outdir", default="plots" ,help="Path of the output folder")
parser.add_argument("--taskname", default="stack_blocks_sac" ,help="Task name")
parser.add_argument("--sim2sim", type=bool, default=False ,help="Task name")
parser.add_argument("--test2env", default="simsimv2" ,help="Name of test2 env")
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
exps.append({"name": ["0326_B"], "seed_num":[3], "color": ["orange","orange"], "plotname": "1 agent fb: False; uf: 2.0; alpha: 0.1"})
exps.append({"name": ["0325_D"], "seed_num":[3], "color": ["lime","lime"], "plotname": "1 agent fb: False; uf: 2.0; alpha: 0.2"})
exps.append({"name": ["0326_A"], "seed_num":[3], "color": ["green","green"], "plotname": "1 agent fb: True; uf: 2.0; alpha: 0.2"})
exps.append({"name": ["0326_C"], "seed_num":[3], "color": ["blue","blue"], "plotname": "1 agent fb: False; uf: 2.0; alpha: 0.3"})
exps.append({"name": ["0326_D"], "seed_num":[3], "color": ["magenta","magenta"], "plotname": "1 agent fb: False; uf: 2.0; alpha: 0.4"})

# DEMO
# exps.append({"name": ["0325_D"], "seed_num":[3], "color": ["orange","orange"], "plotname": "1 agent; 0325_D"})
# exps.append({"name": ["0326_E_01","0326_E_02"], "seed_num":[3,3], "color": ["green","green"], "plotname": "1 agent; 0326_E"})
# exps.append({"name": ["0326_F_01","0326_F_02"], "seed_num":[3,3], "color": ["blue","blue"], "plotname": "1 agent; 0326_F"})
# exps.append({"name": ["0326_G_01","0326_G_02"], "seed_num":[3,3], "color": ["magenta","magenta"], "plotname": "1 agent; 0326_G"})

# DEMO MODE
# exps.append({"name": ["0420_A"], "seed_num":[1], "color": ["orange","red"], "plotname": "1 agent; 0420_A"})
# exps.append({"name": ["0420_B"], "seed_num":[3], "color": ["green","lime"], "plotname": "1 agent; 0420_B"})
# exps.append({"name": ["0420_C"], "seed_num":[3], "color": ["blue","cyan"], "plotname": "1 agent; 0420_C"})

# MULTI-AGENT
# exps.append({"name": ["05_15_A"], "seed_num":[3], "color": ["violet","violet"], "plotname": "3 agents 3 buffers; 0515_A"})
# exps.append({"name": ["0516_A"], "seed_num":[3], "color": ["blue","cyan"], "plotname": "1 agent + fallback; 0516_A"})
#exps.append({"name": ["0517_A"], "seed_num":[3], "color": ["green","lime"], "plotname": "1 agent; 0517_A"})

# exps.append({"name": ["0518_A"], "seed_num":[3], "color": ["lime","lime"], "plotname": "1 agent; 0518_A"})
# exps.append({"name": ["0518_B"], "seed_num":[3], "color": ["cyan","cyan"], "plotname": "1 agent + fallback; 0518_B"})
# exps.append({"name": ["0518_C"], "seed_num":[3], "color": ["magenta","magenta"], "plotname": "3 agents 3 buffers; ; 0518_C"})

# exps.append({"name": ["X_0515_A_test"], "seed_num":[1], "color": "blue", "plotname": "X_0515_A_test"})
# exps.append({"name": ["X_0515_B_test"], "seed_num":[1], "color": "orange", "plotname": "X_0515_B_test"})
# exps.append({"name": ["X_0515_C_test"], "seed_num":[1], "color": "purple", "plotname": "X_0515_C_test"})

# exps.append({"name": ["0325_C"], "seed_num":[3], "color": "orange", "plotname": "1 agent; uf = 0.5" })
# exps.append({"name": ["0323_B","0323_C","0324_D"], "seed_num":[3,3,3], "color": "cyan", "plotname": "1 agent; uf=1.0"})
# exps.append({"name": ["0323_A","0324_C"], "seed_num":[3,3], "color": "blue", "plotname": "1 agent; fallback: True; uf=1.0"})

#exps.append({"name": ["0420_A"], "seed_num":[3], "color": "orange", "plotname": "1 agent; 0420_A"})

## SIM2SIM

exp_test_color_list = []
exp_test2_color_list = []
for i in range(len(exps)):
    exp_test_color_list.append(exps[i]['color'][0])
    exp_test2_color_list.append(exps[i]['color'][1])

test_ret = pd.DataFrame({'Step' : [], 'ExpName' : [],'Seed' : [], 'Type' : [], 'Value' : []})

if args.sim2sim:
    for i in range(len(exps)):
        running_id = 0
        for j in range(len(exps[i]['seed_num'])):
            for k in range(exps[i]['seed_num'][j]):
                graph_csv_name = "test_epochs_" + exps[i]['name'][j] +  "_" + str(k) + "_" + args.test2env + ".csv"
                path = os.path.join(current_dir, args.logdir,graph_csv_name)
                pivot = pd.read_csv(path)

                pivot = pivot.drop(columns=['avg_episode_len','error_in_env','out_of_bounds','succes_rate'])     
                

                pivot['ExpName'] = exps[i]['plotname']     
                pivot['Seed'] = str(running_id)

                pivot['Step'] = pivot.index * 10
                pivot = pivot.rename(columns={"avg_return": "Value"})      

                pivot['Type'] = "test2"

                running_id += 1

                #print(pivot)
                test_ret = test_ret.append(pivot, ignore_index = True)

print(test_ret.to_string())
print(test_ret.head())        
         
print(exps)



graph_name = "test_glob_checkpoint_test_return"

#test_ret = pd.DataFrame({'Step' : [], 'ExpName' : [],'Seed' : [],'Value' : []})

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

            pivot['Type'] = "test"

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
            pivot=pivot[pivot["Type"] == "test"] 
            
            if running_id == 0:
                plt.plot(pivot['Step'],pivot['Value'],color=exps[i]['color'][0],label=exps[i]['plotname'])
            else:
                plt.plot(pivot['Step'],pivot['Value'],color=exps[i]['color'][0])

            if args.sim2sim:
                pivot=test_ret[test_ret["ExpName"] == exps[i]['plotname']]  
                pivot=pivot[pivot["Seed"] == str(running_id)]
                pivot=pivot[pivot["Type"] == "test2"] 

                if running_id == 0:
                    plt.plot(pivot['Step'],pivot['Value'],color=exps[i]['color'][1],label=exps[i]['plotname'] + " test 2")
                else:
                    plt.plot(pivot['Step'],pivot['Value'],color=exps[i]['color'][1])
            
            running_id += 1

plt.legend(title='Labels', bbox_to_anchor=(1, 1.01), loc='upper left')
plt.xlabel("Step (x1000)")
plt.ylabel("Average test return")
plt.title(args.title)
figname = args.plotid + '_test_res_separate.png'
plt.savefig(os.path.join(current_dir, args.outdir, figname), bbox_inches='tight')
plt.show()

# SD plot ###################################

fig, _ = plt.subplots(figsize=(10,8))

test_ret_test = test_ret[test_ret["Type"] == "test"] 
sns.lineplot(data=test_ret_test, x="Step", y="Value", hue="ExpName", errorbar=('ci', 50), palette=exp_test_color_list)

test_ret_test2 = test_ret[test_ret["Type"] == "test2"] 
test_ret_test2["ExpName"] += " test2"
sns.lineplot(data=test_ret_test2, x="Step", y="Value", hue="ExpName", errorbar=('ci', 50), palette=exp_test2_color_list)

plt.legend(title='Labels', bbox_to_anchor=(1, 1.01), loc='upper left')
plt.xlabel("Step (x1000)")
plt.ylabel("Average test return")
plt.title(args.title + " with sd")
figname = args.plotid + '_test_res_std.png'
plt.savefig(os.path.join(current_dir, args.outdir, figname),bbox_inches='tight')
plt.show()

