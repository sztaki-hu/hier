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

sns.set_theme()

def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(path + ' folder is created!')
    else:
        print(path + ' folder already exists!')

parser = argparse.ArgumentParser()
parser.add_argument("--outdir", default="results/output/tables" ,help="Path of the output folder")
parser.add_argument("--show", default=False ,help="Id of plot")
args = parser.parse_args()

plotid = 'plotid'
seednum = 10
alg = 'sac' # 'td3','ddpg'
plotdata = 'eval_success_rate'
taskname_list = ['Slide']
taskname_missing_list = ['Push','PickAndPlace']

# taskname_list = ['Push','Slide','PickAndPlace']
# taskname_missing_list = []

create_folder(os.path.join(current_dir, args.outdir, plotid))

dtypes = np.dtype(
    [
        ("exp_name", str),
        ("seed", int),
        ("t", int),
        ("value", float),
        ("plot_name", str),
    ]
)

maxdtypes = np.dtype(
    [
        ("exp_name", str),
        ("maxvalue", float),
        ("meanmaxvalue", float),
        ("stdmaxvalue", float),
        ("plot_name", str),
    ]
)

maxdata_all_pd = None


for task_index in range(len(taskname_list)):
    taskname = taskname_list[task_index]
    
    exps = []

    # SOTA ######################x

    ises = ['max','selfpaced']
    hers = ['noher','final']
    hiers = ['nohier','predefined']
    pers = ['noper','proportional']

    datetag = '1116' if taskname in ['Push','Slide'] else '1119'
    
    idx = 0
    for hier in hiers:
        for ise in ises:
            for per in pers:
                for her in hers:
                    if per == 'noper':
                        exps.append({"exp_name": "_".join([datetag,'A',alg,ise,her,hier,'fix',per,'sparse','Panda'+taskname+'-v3']) , "seed_num":seednum, "plot_name": " + ".join([her,per,ise,hier])}) # type: ignore
                    else:
                        exps.append({"exp_name": "_".join([datetag,'A',alg,ise,her,hier,'prioritized',per,'sparse','Panda'+taskname+'-v3']) , "seed_num":seednum, "plot_name": " + ".join([her,per,ise,hier])}) # type: ignore
                    idx += 1



  

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


    maxdata_pd = pd.DataFrame(np.empty(0, dtype=maxdtypes))

    for i in range(len(exps)):
        maxVals = np.zeros(exps[i]['seed_num'])
        for j in range(exps[i]['seed_num']):
            pivot=data_pd[data_pd["exp_name"] == exps[i]['exp_name']]
            pivot=pivot[pivot["seed"] == j]

            maxVals[j] = pivot['value'].max()

        print(maxVals)
        #assert False
        maxmaxVal = np.max(maxVals)
        meanmaxVal = np.mean(maxVals)
        stdmaxVal = np.std(maxVals)

        df_append = {'exp_name': exps[i]['exp_name'], 
                        'maxvalue': maxmaxVal,
                        'meanmaxvalue': meanmaxVal,
                        'stdmaxvalue': stdmaxVal,
                        'plot_name': pivot.iloc[0]["plot_name"]}
        maxdata_pd = maxdata_pd.append(df_append, ignore_index = True)

    if task_index == 0:
        maxdata_all_pd = maxdata_pd
    else:
        maxdata_all_pd['maxvalue_'+str(task_index)] = maxdata_pd['maxvalue']
        maxdata_all_pd['meanmaxvalue_'+str(task_index)] = maxdata_pd['meanmaxvalue']
        maxdata_all_pd['stdmaxvalue_'+str(task_index)] = maxdata_pd['stdmaxvalue']

for _ in range(len(taskname_missing_list)):
    task_index += 1
    maxdata_all_pd['maxvalue_'+str(task_index)] = "-"
    maxdata_all_pd['meanmaxvalue_'+str(task_index)] = "-"
    maxdata_all_pd['stdmaxvalue_'+str(task_index)] = "-"
    


maxdata_all_pd['HER'] = "-"
maxdata_all_pd['PER'] = "-"
maxdata_all_pd['ISE'] = "-"
maxdata_all_pd['HiER'] = "-"

for ind in maxdata_all_pd.index:
    if maxdata_all_pd['exp_name'][ind].find('noher') == -1: maxdata_all_pd['HER'][ind] = "\checkmark"
    if maxdata_all_pd['exp_name'][ind].find('noper') == -1: maxdata_all_pd['PER'][ind] = "\checkmark"
    if maxdata_all_pd['exp_name'][ind].find('max') == -1: maxdata_all_pd['ISE'][ind] = "\checkmark"
    if maxdata_all_pd['exp_name'][ind].find('nohier') == -1: maxdata_all_pd['HiER'][ind] = "\checkmark"

maxdata_all_pd = maxdata_all_pd[['HER','PER','ISE','HiER','maxvalue','meanmaxvalue','stdmaxvalue','maxvalue_1','meanmaxvalue_1','stdmaxvalue_1','maxvalue_2','meanmaxvalue_2','stdmaxvalue_2']]

print(maxdata_all_pd.to_string())
print(maxdata_all_pd.head())


s = maxdata_all_pd.to_latex(index=False,
            formatters={"name": str.upper},
            float_format="{:.2f}".format,
            escape=False)


with open(os.path.join(current_dir, args.outdir, plotid, plotdata+"_latex.txt"), "w") as text_file:
    text_file.write(s)

   


