import os
from os.path import dirname, abspath
current_dir = dirname(abspath(__file__))
current_dir = os.path.abspath(os.path.join(current_dir, os.pardir))

import numpy as np
import pandas as pd
import argparse

# Import seaborn
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

sns.set_theme()

def create_folder(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)
        print(path + ' folder is created!')
    else:
        print(path + ' folder already exists!')

parser = argparse.ArgumentParser()
#parser.add_argument("--plotid", default="1004_A_slide" ,help="Id of plot")
parser.add_argument("--outdir", default="results/output/plots" ,help="Path of the output folder")
parser.add_argument("--show", default=False ,help="-")
parser.add_argument("--createcsv", default=True ,help="-")
parser.add_argument("--legend", default=True ,help="-")
args = parser.parse_args()

fontsize_title = 32
fontsize_label = 28
fontsize_label_next_figure = 18
fontsize_ticks = 22

fontsize_axis_label = 28
legend_columns = 6
color_line_width = 4.0

linewidth = 2

plotid = '1105_B'
seednum = 3
#plotdata_list = ['eval_success_rate',"rollout_success_rate","cl_ratio","rollout_state_changed","eval_state_change_rate","hl_highlights_buffer_size","time_fps"]
#taskname_list = ['Reach','Push','Slide','PickAndPlace','Stack']
#taskname_list = ['Push']
#taskname_list = ['Push','Slide','PickAndPlace']
#taskname_list = ['Push']
taskname_list = ['Slide']
plotdata_list = ['eval_success_rate','eval_mean_reward',
                 'rollout_success_rate', 'rollout_state_changed','rollout_ep_rew_mean',
                 'cl_c',
                 'time_fps',
                 'hier_batch_size','hier_buffer_size','hier_lambda','hier_xi']
plotdata_y_list = ['Eval success rate','Eval Mean reward',
                 'Rollout success rate', 'Rollout state changed','Rollout ep rew mean',
                 'CL param c',
                 'time fps',
                 'HiER batch size','HiER buffer size','HiER lambda','HiER xi']

# plotdata_list = ['hier_batch_size','hier_buffer_size','hier_lambda','hier_xi']
# plotdata_y_list = ['HiER batch size','HiER buffer size','HiER lambda','HiER xi']


# plotdata_list = ['eval_success_rate','eval_mean_reward']
# plotdata_y_list = ['Eval success rate','Eval sum reward']

# plotdata_list = ['eval_success_rate']
# plotdata_y_list = ['Eval success rate']

#print(os.path.join(current_dir, args.outdir))

algs = ['sac']


for taskname in taskname_list:
    for alg in algs:
    
        logdir = plotid + "_Panda" + taskname + "_" + alg
        create_folder(os.path.join(current_dir, args.outdir, logdir))
        exps = []

        # HiER ###########################################################xxx

        # exps.append({"exp_name": "_".join(['1105_A', 'Panda'+taskname+'-v3',alg,'sparse','final','nohier','fix','noper','5e5','controladaptive']) , "seed_num":seednum, "color": "red", "plot_name":  'HER + CL'})
        # exps.append({"exp_name": "_".join(['1105_A', 'Panda'+taskname+'-v3',alg,'sparse','final','fix','fix','noper','5e5','controladaptive']) , "seed_num":seednum, "color": "orange", "plot_name":  'HER + CL + HiER(fix)'})
        # exps.append({"exp_name": "_".join(['1105_A', 'Panda'+taskname+'-v3',alg,'sparse','final','multifix','fix','noper','5e5','controladaptive']) , "seed_num":seednum, "color": "green", "plot_name":  'HER + CL + HiER(multifix)'})
        # exps.append({"exp_name": "_".join(['1105_A', 'Panda'+taskname+'-v3',alg,'sparse','final','predefined','fix','noper','5e5','controladaptive']) , "seed_num":seednum, "color": "blue", "plot_name":  'HER + CL + HiER(predefined)'})
        # exps.append({"exp_name": "_".join(['1105_A', 'Panda'+taskname+'-v3',alg,'sparse','final','ama','fix','noper','5e5','controladaptive']) , "seed_num":seednum, "color": "magenta", "plot_name":  'HER + CL + HiER(ama)'})
        # exps.append({"exp_name": "_".join(['1105_A', 'Panda'+taskname+'-v3',alg,'sparse','final','amar','fix','noper','5e5','controladaptive']) , "seed_num":seednum, "color": "purple", "plot_name":  'HER + CL + HiER(amar)'})

        # HiERp ###########################################################xxx

        exps.append({"exp_name": "_".join(['1105_A', 'Panda'+taskname+'-v3',alg,'sparse','final','nohier','prioritized','noper','5e5','controladaptive']) , "seed_num":seednum, "color": "red", "plot_name":  'HER + CL'})
        exps.append({"exp_name": "_".join(['1105_A', 'Panda'+taskname+'-v3',alg,'sparse','final','fix','prioritized','noper','5e5','controladaptive']) , "seed_num":seednum, "color": "orange", "plot_name":  'HER + CL + HiER(fix)'})
        exps.append({"exp_name": "_".join(['1105_A', 'Panda'+taskname+'-v3',alg,'sparse','final','multifix','prioritized','noper','5e5','controladaptive']) , "seed_num":seednum, "color": "green", "plot_name":  'HER + CL + HiER(multifix)'})
        exps.append({"exp_name": "_".join(['1105_A', 'Panda'+taskname+'-v3',alg,'sparse','final','predefined','prioritized','noper','5e5','controladaptive']) , "seed_num":seednum, "color": "blue", "plot_name":  'HER + CL + HiER(predefined)'})
        exps.append({"exp_name": "_".join(['1105_A', 'Panda'+taskname+'-v3',alg,'sparse','final','ama','prioritized','noper','5e5','controladaptive']) , "seed_num":seednum, "color": "magenta", "plot_name":  'HER + CL + HiER(ama)'})
        exps.append({"exp_name": "_".join(['1105_A', 'Panda'+taskname+'-v3',alg,'sparse','final','amar','prioritized','noper','5e5','controladaptive']) , "seed_num":seednum, "color": "purple", "plot_name":  'HER + CL + HiER(amar)'})


        for plotdata,plotdata_y in zip(plotdata_list,plotdata_y_list):

            exp_test_color_list = []
            for i in range(len(exps)):                
                if (exps[i]['exp_name'].find('multifix') != -1) and (plotdata in ['hier_batch_size','hier_buffer_size','hier_lambda','hier_xi']):
                    continue
                
                exp_test_color_list.append(exps[i]['color'])

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
                    if (exps[i]['exp_name'].find('multifix') != -1) and (plotdata in ['hier_batch_size','hier_buffer_size','hier_lambda','hier_xi']):
                        continue
                    path = os.path.join(current_dir, "logs",exps[i]['exp_name'],str(j),'runs','csv',plotdata+'.csv')
                    print(path)
                    pivot = pd.read_csv(path)

                    column_names = list(pivot.columns)
                    pivot = pivot.rename(columns={column_names[0]: 't', column_names[1]: "value"})
                    pivot['exp_name'] = exps[i]['exp_name']
                    pivot['plot_name'] = exps[i]['plot_name']
                    pivot['seed'] = int(j)
                    pivot['t'] = pivot['t'] / 1000
                    pivot = pivot[['exp_name','seed','t','value','plot_name']]

                    data_pd = data_pd.append(pivot, ignore_index = True) # type: ignore

                        
            print(data_pd.to_string())
            print(data_pd.head())

            if args.createcsv:
                if plotdata in ['eval_success_rate','eval_mean_reward']:

                    maxdtypes = np.dtype(
                        [
                            ("exp_name", str),
                            ("meanmaxvalue", float),
                            ("stdmaxvalue", float),
                            ("plot_name", str),
                        ]
                    )
                    maxdata_pd = pd.DataFrame(np.empty(0, dtype=maxdtypes))

                    for i in range(len(exps)):
                        maxVals = np.zeros(exps[i]['seed_num'])
                        for j in range(exps[i]['seed_num']):
                            pivot=data_pd[data_pd["exp_name"] == exps[i]['exp_name']]
                            pivot=pivot[pivot["seed"] == j]

                            maxVals[j] = pivot['value'].max()
                        meanmaxVal = np.mean(maxVals)
                        stdmaxVal = np.std(maxVals)

                        df_append = {'exp_name': exps[i]['exp_name'], 
                                        'meanmaxvalue': meanmaxVal,
                                        'stdmaxvalue': stdmaxVal,
                                        'plot_name': pivot.iloc[0]["plot_name"]} # type: ignore
                        maxdata_pd = maxdata_pd.append(df_append, ignore_index = True) # type: ignore
            
                    maxdata_pd['HER'] = False
                    maxdata_pd['PER'] = False
                    maxdata_pd['CL'] = False
                    maxdata_pd['HiER'] = False

                    for ind in maxdata_pd.index:
                        if maxdata_pd['exp_name'][ind].find('noher') == -1: maxdata_pd['HER'][ind] = True
                        if maxdata_pd['exp_name'][ind].find('noper') == -1: maxdata_pd['PER'][ind] = True
                        if maxdata_pd['exp_name'][ind].find('nocl') == -1: maxdata_pd['CL'][ind] = True
                        if maxdata_pd['exp_name'][ind].find('nohier') == -1: maxdata_pd['HiER'][ind] = True
                
                    maxdata_pd = maxdata_pd[['HER','PER','CL','HiER','meanmaxvalue','stdmaxvalue']]

                    print(maxdata_pd.to_string())
                    print(maxdata_pd.head())

                    maxdata_pd.to_csv(os.path.join(current_dir, args.outdir, logdir, plotdata+".csv"),index=False)

            # Separate plotting ########################## 

            fig, ax = plt.subplots(figsize=(10,8))
            #exps_names = data_pd['exp_name'].unique()

            for i in range(len(exps)):
                for j in range(exps[i]['seed_num']):
                    if (exps[i]['exp_name'].find('multifix') != -1) and (plotdata in ['hier_batch_size','hier_buffer_size','hier_lambda','hier_xi']):
                        continue
                    pivot=data_pd[data_pd["exp_name"] == exps[i]['exp_name']]  
                    pivot=pivot[pivot["seed"] == j]
                    if j == 0:
                        plt.plot(pivot['t'],
                                pivot['value'],
                                color=exps[i]['color'],
                                label=exps[i]['plot_name'],
                                linewidth=linewidth)  
                    else: 
                        plt.plot(pivot['t'],
                                pivot['value'],
                                color=exps[i]['color'],
                                linewidth=linewidth)
                        

            if args.legend: 
                legend = plt.legend(bbox_to_anchor=(1, 1.01), loc='upper left', fontsize=fontsize_label_next_figure)
                legend.get_frame().set_facecolor('white')
                for line in legend.get_lines():
                    line.set_linewidth(2.0)

            plt.xlabel("Timesteps (x1000)", fontsize=fontsize_axis_label)
            plt.ylabel(plotdata_y, fontsize=fontsize_axis_label)
            plt.title(taskname, fontsize=fontsize_title)
            plt.xticks(fontsize=fontsize_ticks)
            plt.yticks(fontsize=fontsize_ticks)
            ax.set_facecolor((1.0, 1.0, 1.0))
            ax.spines['bottom'].set_color('black')
            ax.spines['top'].set_color('black') 
            ax.spines['right'].set_color('black')
            ax.spines['left'].set_color('black')
            plt.grid(c='gray')

            figname = plotid + '_' + plotdata
            plt.savefig(os.path.join(current_dir, args.outdir, logdir, figname),bbox_inches='tight')
            if args.show: plt.show()

            plt.clf()
            plt.cla()

            # SD plot ###################################

            fig, ax = plt.subplots(figsize=(10,8))

            sns.lineplot(data=data_pd, 
                         x="t", 
                         y="value", 
                         hue="plot_name", 
                         errorbar=('ci', 95), 
                         palette=exp_test_color_list, 
                         legend=args.legend,
                         linewidth = linewidth)

            if args.legend: 
                legend = plt.legend(bbox_to_anchor=(1, 1.01), loc='upper left', fontsize=fontsize_label_next_figure)
                legend.get_frame().set_facecolor('white')
                for line in legend.get_lines():
                    line.set_linewidth(2.0)
            plt.xlabel("Timesteps")
            plt.ylabel(plotdata_y)
            plt.xlabel("Timesteps (x1000)", fontsize=fontsize_axis_label)
            plt.ylabel(plotdata_y, fontsize=fontsize_axis_label)
            plt.title(taskname, fontsize=fontsize_title)
            plt.xticks(fontsize=fontsize_ticks)
            plt.yticks(fontsize=fontsize_ticks)
            ax.set_facecolor((1.0, 1.0, 1.0))
            ax.spines['bottom'].set_color('black')
            ax.spines['top'].set_color('black') 
            ax.spines['right'].set_color('black')
            ax.spines['left'].set_color('black')
            plt.grid(c='gray')

            figname = plotid + '_'+plotdata+'_std.png'
            plt.savefig(os.path.join(current_dir, args.outdir, logdir, figname),bbox_inches='tight')
            if args.show: plt.show()

            plt.clf()
            plt.cla()

            # Legend ########################## 

            fig, ax = plt.subplots(figsize=(10,8))
            #exps_names = data_pd['exp_name'].unique()

            for i in range(len(exps)):
                for j in range(exps[i]['seed_num']):
                    pivot=data_pd[data_pd["exp_name"] == exps[i]['exp_name']]  
                    pivot=pivot[pivot["seed"] == j]
                    plt.plot(pivot['t'],pivot['value'],color=exps[i]['color'],label=exps[i]['plot_name']) if j == 0 else plt.plot(pivot['t'],pivot['value'],color=exps[i]['color'])
                    

            legend = plt.legend(bbox_to_anchor=(0.5, -0.2), loc='upper center', 
                        ncol=legend_columns, frameon=False, fontsize=fontsize_label)
            legend.get_frame().set_facecolor('white')
            for line in legend.get_lines():
                line.set_linewidth(color_line_width)

            def export_legend(legend, expand=[-5,-5,5,5]):
                fig  = legend.figure
                fig.canvas.draw()
                bbox  = legend.get_window_extent()
                bbox = bbox.from_extents(*(bbox.extents + np.array(expand)))
                bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
                fig.savefig(os.path.join(current_dir, args.outdir, logdir, "legend.png"), dpi="figure", bbox_inches=bbox, facecolor='white')

            export_legend(legend)

            plt.clf()
            plt.cla()

