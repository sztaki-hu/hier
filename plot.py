import os
from os.path import dirname, abspath
current_dir = dirname(abspath(__file__))

import numpy as np
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
parser.add_argument("--outdir", default="plots" ,help="Path of the output folder")
parser.add_argument("--show", default=False ,help="Id of plot")
args = parser.parse_args()

plotdata_list = ['eval_success_rate',"rollout_success_rate","cl_ratio"]

create_folder(os.path.join(current_dir, args.outdir))

exps = []

# General
exp_id = "0920_X2"
taskname = 'Push'
her = "final"
exps.append({"exp_name": exp_id+"_Panda"+taskname+"-v3_sac_"+her+"_linear", "seed_num":3, "color": "blue", "plot_name": "linear"})
exps.append({"exp_name": exp_id+"_Panda"+taskname+"-v3_sac_"+her+"_selfpaced", "seed_num":3, "color": "green", "plot_name": "selfpaced"})
exps.append({"exp_name": exp_id+"_Panda"+taskname+"-v3_sac_"+her+"_nocl", "seed_num":3, "color": "purple", "plot_name": "nocl"})

exp_test_color_list = []
for i in range(len(exps)):
    exp_test_color_list.append(exps[i]['color'])

for plotdata in plotdata_list:

    dtypes = np.dtype(
        [
            ("exp_name", str),
            ("seed", int),
            ("t", int),
            ("value", float),
            ("plot_name", str),
        ]
    )
    data_pd = pd.DataFrame(np.empty(0, dtype=dtypes))
    print(data_pd.dtypes)

    for i in range(len(exps)):
        running_id = 0
        for j in range(exps[i]['seed_num']):
            path = os.path.join(current_dir, "logs",exps[i]['exp_name'],str(j),'runs','csv',plotdata+'.csv')
            print(path)
            pivot = pd.read_csv(path)

            column_names = list(pivot.columns)
            pivot = pivot.rename(columns={column_names[0]: 't', column_names[1]: "value"})
            pivot['exp_name'] = exps[i]['exp_name']
            pivot['plot_name'] = exps[i]['plot_name']
            pivot['seed'] = int(j)
            pivot = pivot[['exp_name','seed','t','value','plot_name']]

            data_pd = data_pd.append(pivot, ignore_index = True)

                
    print(data_pd.to_string())
    print(data_pd.head())

    # Separate plotting ########################## 

    fig, _ = plt.subplots(figsize=(10,8))
    #exps_names = data_pd['exp_name'].unique()

    for i in range(len(exps)):
        for j in range(exps[i]['seed_num']):
            pivot=data_pd[data_pd["exp_name"] == exps[i]['exp_name']]  
            pivot=pivot[pivot["seed"] == j]
            plt.plot(pivot['t'],pivot['value'],color=exps[i]['color'],label=exps[i]['plot_name']) if j == 0 else plt.plot(pivot['t'],pivot['value'],color=exps[i]['color'])
            

    plt.legend(title='Labels', bbox_to_anchor=(1, 1.01), loc='upper left')
    plt.xlabel("t")
    plt.ylabel(plotdata)
    plt.title(plotdata)
    figname = args.plotid + "_" + plotdata + '_.png'
    plt.savefig(os.path.join(current_dir, args.outdir, figname), bbox_inches='tight')
    if args.show: plt.show()

    # SD plot ###################################

    fig, _ = plt.subplots(figsize=(10,8))

    sns.lineplot(data=data_pd, x="t", y="value", hue="plot_name", errorbar=('ci', 95), palette=exp_test_color_list)

    plt.legend(title='Labels', bbox_to_anchor=(1, 1.01), loc='upper left')
    plt.xlabel("t")
    plt.ylabel(plotdata)
    plt.title(plotdata + " with sd")
    figname = args.plotid + '_'+plotdata+'_std.png'
    plt.savefig(os.path.join(current_dir, args.outdir, figname),bbox_inches='tight')
    if args.show: plt.show()

