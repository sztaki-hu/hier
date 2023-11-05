import os
from os.path import dirname, abspath
current_dir = dirname(abspath(__file__))
current_dir = os.path.abspath(os.path.join(current_dir, os.pardir))

import numpy as np
import pandas as pd
import argparse

# Import seaborn
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib

sns.set_theme()

def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(path + ' folder is created!')
    else:
        print(path + ' folder already exists!')

parser = argparse.ArgumentParser()
#parser.add_argument("--plotid", default="1004_A_slide" ,help="Id of plot")
parser.add_argument("--outdir", default="results/output/plots_multifix" ,help="Path of the output folder")
parser.add_argument("--show", default=False ,help="Id of plot")
args = parser.parse_args()


plotid = '1105_X'
seednum = 1
taskname_list = ['Push']
#taskname_list = ['InvertedPendulum-v4']


create_folder(os.path.join(current_dir, args.outdir))

algs = ['sac']

runtime_cat_name_list = ['0', '1', '2', '3']
exp_test_color_list = ['green','orange','blue','magenta'] # collect, process ep, train, test, other

for taskname in taskname_list:
    for alg in algs:
        
        exps = []
       
        # exps.append({"exp_name": "_".join(['1019_A', 'Panda'+taskname+'-v3',alg,'sparse','final','multifix','fix','noper','controldiscreteadaptive']) , "seed_num":seednum, "color": "purple", "plot_name":  alg +" HER+HiER(multifix)+CL"})
        # exps.append({"exp_name": "_".join(['1019_A', 'Panda'+taskname+'-v3',alg,'sparse','final','multifix','prioritized','noper','controldiscreteadaptive']) , "seed_num":seednum, "color": "magenta", "plot_name":  alg +" HER+PHiER(multifix)+CL"})
        
        exps.append({"exp_name": "_".join(['1105_T2', 'Panda'+taskname+'-v3',alg,'sparse','final','multifix','prioritized','noper','5e4','controladaptive']) , "seed_num":seednum, "color": "purple", "plot_name":  'multifix'})


        for exp in exps:

            logdir = "_".join([plotid, taskname, alg])
            create_folder(os.path.join(current_dir, args.outdir, logdir))

            dtypes = np.dtype(
                [
                    ('exp_name', str),
                    ('t', int),
                    ('category',str),
                    ('value', float)
                ]
            )
            data_pd = pd.DataFrame(np.empty(0, dtype=dtypes))
            print(data_pd.dtypes)

            runtime_list = []
            for cat_name in runtime_cat_name_list:
                values = []
                for j in range(seednum):
                    read_pd = pd.read_csv(os.path.join(current_dir, "logs",exp['exp_name'],str(j),'runs','csv','hier_batch_size_'+cat_name+'.csv'))         
                    if j == 0 : 
                        values = read_pd.iloc[:, 1]
                    else:
                        values += read_pd.iloc[:, 1]
                
                values = values / seednum

                pivot = pd.DataFrame(np.empty(0, dtype=dtypes))
                pivot['t'] = read_pd.iloc[:, 0]
                pivot['value'] = values if cat_name == runtime_cat_name_list[0] else values + sum_list
                pivot['exp_name'] = exp['exp_name']
                pivot['category'] = cat_name

                sum_list = pivot['value'] 
            
                data_pd = data_pd.append(pivot, ignore_index = True)        

            print(data_pd.to_string())
            print(data_pd.head())

            # Separate plotting ########################## 

            fig, ax = plt.subplots(figsize=(10,8))
        
            for name,color in zip(runtime_cat_name_list,exp_test_color_list):
                pivot=data_pd[data_pd["category"] == name]
                plt.plot(pivot['t'],pivot['value'],color=color,label=name)
            
            pivot=data_pd[data_pd["category"] == runtime_cat_name_list[0]]
            ax.fill_between(pivot['t'], pivot['value'], facecolor=exp_test_color_list[0], alpha=0.5)
            #ax.fill_between(pivot['t'], pivot['value'], facecolor=exp_test_color_list[0])

            for i in range(len(runtime_cat_name_list)-1):
                pivot_1=data_pd[data_pd["category"] == runtime_cat_name_list[i+1]]
                pivot_2=data_pd[data_pd["category"] == runtime_cat_name_list[i]]
                ax.fill_between(pivot['t'], pivot_1['value'], pivot_2["value"], facecolor=exp_test_color_list[i+1], alpha=0.5)
                #ax.fill_between(pivot['t'], pivot['value'], pivot_2["value"], facecolor=exp_test_color_list[i+1])

            plt.legend(title='Labels', bbox_to_anchor=(1, 1.01), loc='upper left')
            plt.xlabel("t (step)")
            plt.ylabel("Share")
            plt.title("Batch sizes " + exp["plot_name"])
            figname = "_".join([plotid,exp["plot_name"],"runtime_analysis.png"])
            plt.savefig(os.path.join(current_dir, args.outdir, logdir, figname), bbox_inches='tight')
            if args.show: plt.show()


