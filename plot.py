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
#parser.add_argument("--plotid", default="1004_A_slide" ,help="Id of plot")
parser.add_argument("--outdir", default="plots" ,help="Path of the output folder")
parser.add_argument("--show", default=False ,help="Id of plot")
args = parser.parse_args()


plotid = '1015_A'
seednum = 3
plotdata_list = ['eval_success_rate',"rollout_success_rate","cl_ratio","rollout_state_changed","eval_state_change_rate","hl_highlights_buffer_size"]
taskname_list = ['Reach','Push','Slide','PickAndPlace','Stack']
#taskname_list = ['Slide']
#taskname_list = ['PickAndPlace']

create_folder(os.path.join(current_dir, args.outdir))

algs = ['sac','td3','ddpg']


for taskname in taskname_list:
    for alg in algs:
    
        logdir = plotid + "_Panda" + taskname + "_" + alg
        create_folder(os.path.join(current_dir, args.outdir, logdir))
        exps = []

        
        exps.append({"exp_name": "_".join(['1012_B', 'Panda'+taskname+'-v3',alg,'sparse','noher','nohl','noper','nocl']) , "seed_num":seednum, "color": "brown", "plot_name": alg})
        exps.append({"exp_name": "_".join(['1012_B', 'Panda'+taskname+'-v3',alg,'sparse','final','nohl','noper','nocl']) , "seed_num":seednum, "color": "orange", "plot_name": alg + " HER"})
        exps.append({"exp_name": "_".join(['1012_B', 'Panda'+taskname+'-v3',alg,'sparse','noher','nohl','proportional','nocl']) , "seed_num":seednum, "color": "olive", "plot_name":  alg +" PER"})
        exps.append({"exp_name": "_".join(['1012_B', 'Panda'+taskname+'-v3',alg,'sparse','final','nohl','proportional','nocl']) , "seed_num":seednum, "color": "green", "plot_name":  alg +" HER+PER"})
        #exps.append({"exp_name": "_".join(['1012_B', 'Panda'+taskname+'-v3',alg,'sparse','final','fix','proportional','controldiscreteadaptive']) , "seed_num":seednum, "color": "blue", "plot_name":  alg +" HER+PER+HiER(fix)+CL"})
        #exps.append({"exp_name": "_".join(['1012_B', 'Panda'+taskname+'-v3',alg,'sparse','final','ama','proportional','controldiscreteadaptive']) , "seed_num":seednum, "color": "navy", "plot_name":  alg +" HER+PER+HiER(ama)+CL"})
        exps.append({"exp_name": "_".join(['1015_A', 'Panda'+taskname+'-v3',alg,'sparse','noher','ama','noper','nocl']) , "seed_num":seednum, "color": "aqua", "plot_name":  alg +" HiER"})
        exps.append({"exp_name": "_".join(['1015_A', 'Panda'+taskname+'-v3',alg,'sparse','noher','nohl','noper','controldiscreteadaptive']) , "seed_num":seednum, "color": "magenta", "plot_name":  alg +" CL"})

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
            figname = plotid + "_" + plotdata + '_.png'
            plt.savefig(os.path.join(current_dir, args.outdir, logdir, figname), bbox_inches='tight')
            if args.show: plt.show()

            # SD plot ###################################

            fig, _ = plt.subplots(figsize=(10,8))

            sns.lineplot(data=data_pd, x="t", y="value", hue="plot_name", errorbar=('ci', 95), palette=exp_test_color_list)

            plt.legend(title='Labels', bbox_to_anchor=(1, 1.01), loc='upper left')
            plt.xlabel("t")
            plt.ylabel(plotdata)
            plt.title(plotdata + " with sd")
            figname = plotid + '_'+plotdata+'_std.png'
            plt.savefig(os.path.join(current_dir, args.outdir, logdir, figname),bbox_inches='tight')
            if args.show: plt.show()

