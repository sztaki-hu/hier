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

fontsize_label_next_figure = 18
fontsize_ticks = 22

fontsize_axis_label = 28

fontsize_label = 28 # 28
legend_columns = 5
color_line_width = 6.0

linewidth = 3

plotid = '0207_B_7x7_S'
seednum = 10
#taskname_list = ['Push','Slide','PickAndPlace','Stack']
taskname_list = ['PointMaze_UMaze-v3'] #['PointMaze_UMaze-v3','AntMaze_UMaze-v4', 'AntMaze_UMazeDense-v4']
#taskname_list = ['FetchPush-v2','FetchSlide-v2','FetchPickAndPlace-v2']

plotdata_list = ['eval_success_rate','eval_mean_reward',
                 'rollout_success_rate', 'rollout_state_changed','rollout_ep_rew_mean',
                 'ise_c',
                 'time_fps',
                 'hier_batch_size','hier_buffer_size','hier_lambda','hier_xi']
plotdata_y_list = ['Eval success rate','Eval Mean reward',
                 'Rollout success rate', 'Rollout state changed','Rollout ep rew mean',
                 'ISE param c',
                 'time fps',
                 'HiER batch size','HiER buffer size','HiER lambda','HiER xi']


# Grand test ###################

# plotdata_list = ['eval_success_rate']
# plotdata_y_list = ['Eval success rate']

# HiER #########################

# plotdata_list = ['eval_success_rate','hier_lambda']
# plotdata_y_list = ['Eval success rate','HiER lambda']

# plotdata_list = ['hier_lambda']
# plotdata_y_list = ['HiER lambda']

# E2H ############################

# plotdata_list = ['eval_success_rate','rollout_success_rate','ise_c']
# plotdata_y_list = ['Eval success rate','Rollout success rate','ISE param c']

# plotdata_list = ['ise_c']
# plotdata_y_list = ['ISE param c']



#print(os.path.join(current_dir, args.outdir))

algs = ['sac']
#algs = ['ddpg','td3']


for taskname in taskname_list:

    if taskname == 'PickAndPlace':
        taskname_title = 'Pick-and-place'
    else:
        taskname_title = taskname

    for alg in algs:
    
        logdir = plotid + "_" + taskname + "_" + alg
        create_folder(os.path.join(current_dir, args.outdir, logdir))
        exps = []
    
        color_palette = ['gray','orange','red','green','blue','aqua','magenta','purple']
        color_index = 0

        hers = ['noher','final']
        hiers = ['nohier','predefined']
        ises = ['max']

        # hers = ['final']
        # hiers = ['predefined']
        # ises = ['max']

        for her in hers:
            for hier in hiers:
                for ise in ises:
                    # letter = 'C' if taskname == 'FetchPush-v2' and her == 'final' and hier == 'predefined' else 'A'
                    # exps.append({"exp_name": "_".join(['0207',letter,alg,her,hier,'fix',ise,'noper','sparse',taskname]) , "seed_num":seednum, "color": color_palette[color_index], "plot_name": "_".join([her,hier,ise]) , "linestyle": 'solid'}) # type: ignore
                    exps.append({"exp_name": "_".join(['0207_B',alg,her,hier,'fix',ise,'7x7_S','noper','sparse',taskname]) , "seed_num":seednum, "color": color_palette[color_index], "plot_name": "_".join([her,hier,ise]) , "linestyle": 'solid'}) # type: ignore
                    color_index += 1

########### Grand test
        #color_palette = ['darkgray','dimgray','green','blue','purple','magenta']
        #color_palette = [(0.5,0.5,0.5),(0.2,0.2,0.2),'green','blue','purple','magenta']

        # exps.append({"exp_name": "_".join(['1109_A',alg,'nocl','noher','nohier','fix','noper','sparse','Panda'+taskname+'-v3']) , "seed_num":seednum, "color": color_palette[0], "plot_name": "Baseline"})
        # exps.append({"exp_name": "_".join(['1109_A',alg,'nocl','final','nohier','prioritized','proportional','sparse','Panda'+taskname+'-v3']) , "seed_num":seednum, "color": color_palette[1], "plot_name": "Baseline (HER + PER)"}) 
        # exps.append({"exp_name": "_".join(['1109_A',alg,'selfpaced','final','nohier','prioritized','proportional','sparse','Panda'+taskname+'-v3']) , "seed_num":seednum, "color": color_palette[2], "plot_name": "HER + PER + CL"}) 
        # exps.append({"exp_name": "_".join(['1109_A',alg,'nocl','final','predefined','prioritized','proportional','sparse','Panda'+taskname+'-v3']) , "seed_num":seednum, "color": color_palette[3], "plot_name": "HER + PER + HiER"}) 
        # exps.append({"exp_name": "_".join(['1109_A',alg,'selfpaced','final','predefined','prioritized','proportional','sparse','Panda'+taskname+'-v3']) , "seed_num":seednum, "color": color_palette[4], "plot_name": "HER + PER + HiERCuLe"}) 
        # exps.append({"exp_name": "_".join(['1109_A',alg,'selfpaced','final','predefined','fix','noper','sparse','Panda'+taskname+'-v3']) , "seed_num":seednum, "color": color_palette[5], "plot_name": "HER + HiERCuLe"}) 

        # exps.append({"exp_name": "_".join(['1108_A',alg,'nocl','noher','nohier','fix','noper','sparse','Panda'+taskname+'-v3']) , "seed_num":seednum, "color": color_palette[0], "plot_name": "Baseline"})
        # exps.append({"exp_name": "_".join(['1108_B',alg,'nocl','final','nohier','prioritized','proportional','sparse','Panda'+taskname+'-v3']) , "seed_num":seednum, "color": color_palette[1], "plot_name": "Baseline (with HER & PER)"}) 
        # exps.append({"exp_name": "_".join(['1108_B',alg,'selfpaced','final','nohier','prioritized','proportional','sparse','Panda'+taskname+'-v3']) , "seed_num":seednum, "color": color_palette[2], "plot_name": "E2H-ISE (with HER & PER)"}) 
        # exps.append({"exp_name": "_".join(['1108_B',alg,'nocl','final','predefined','prioritized','proportional','sparse','Panda'+taskname+'-v3']) , "seed_num":seednum, "color": color_palette[3], "plot_name": "HiER (with HER & PER)"}) 
        # exps.append({"exp_name": "_".join(['1108_B',alg,'selfpaced','final','predefined','prioritized','proportional','sparse','Panda'+taskname+'-v3']) , "seed_num":seednum, "color": color_palette[4], "plot_name": "HiER+ (with HER & PER)"}) 
        # exps.append({"exp_name": "_".join(['1108_A',alg,'selfpaced','final','predefined','fix','noper','sparse','Panda'+taskname+'-v3']) , "seed_num":seednum, "color": color_palette[5], "plot_name": "HiER+ (with HER)"}) 
      
        # datetag = '1116' if taskname in ['Push','Slide'] else '1119'
        
        # exps.append({"exp_name": "_".join([datetag,'A',alg,'nocl','noher','nohier','fix','noper','sparse','Panda'+taskname+'-v3']) , "seed_num":seednum, "color": color_palette[0], "plot_name": "Baseline", "linestyle": 'dashed'})
        # exps.append({"exp_name": "_".join([datetag,'A',alg,'nocl','final','nohier','prioritized','proportional','sparse','Panda'+taskname+'-v3']) , "seed_num":seednum, "color": color_palette[1], "plot_name": "Baseline [HER & PER]", "linestyle": 'dashed' }) 
        # exps.append({"exp_name": "_".join([datetag,'A',alg,'selfpaced','final','nohier','prioritized','proportional','sparse','Panda'+taskname+'-v3']) , "seed_num":seednum, "color": color_palette[2], "plot_name": "E2H-ISE [HER & PER]", "linestyle": 'solid' }) 
        # exps.append({"exp_name": "_".join([datetag,'A',alg,'nocl','final','predefined','prioritized','proportional','sparse','Panda'+taskname+'-v3']) , "seed_num":seednum, "color": color_palette[3], "plot_name": "HiER [HER & PER]", "linestyle": 'solid' }) 
        # exps.append({"exp_name": "_".join([datetag,'A',alg,'selfpaced','final','predefined','prioritized','proportional','sparse','Panda'+taskname+'-v3']) , "seed_num":seednum, "color": color_palette[4], "plot_name": "HiER+ [HER & PER]", "linestyle": 'solid' }) 
        # exps.append({"exp_name": "_".join([datetag,'A',alg,'selfpaced','final','predefined','fix','noper','sparse','Panda'+taskname+'-v3']) , "seed_num":seednum, "color": color_palette[5], "plot_name": "HiER+ [HER]", "linestyle": 'solid' }) 

########### HiER lambda

        # color_palette = [(0.2,0.2,0.2),'orange','blue','purple']
    
        # exps.append({"exp_name": "_".join(['1127_D',alg,'nocl','noher','nohier','fix','noper','sparse','Panda'+taskname+'-v3']) , "seed_num":seednum, "color": color_palette[0], "plot_name": "without", "linestyle": 'dashed'}) # type: ignore
        # exps.append({"exp_name": "_".join(['1127_D',alg,'nocl','noher','fix','fix','noper','sparse','Panda'+taskname+'-v3']) , "seed_num":seednum, "color": color_palette[1], "plot_name": "fix", "linestyle": 'solid'}) 
        # exps.append({"exp_name": "_".join(['1127_D',alg,'nocl','noher','predefined','fix','noper','sparse','Panda'+taskname+'-v3']) , "seed_num":seednum, "color": color_palette[2], "plot_name": "predefined", "linestyle": 'solid'}) 
        # exps.append({"exp_name": "_".join(['1127_D',alg,'nocl','noher','ama','fix','noper','sparse','Panda'+taskname+'-v3']) , "seed_num":seednum, "color": color_palette[3], "plot_name": "ama", "linestyle": 'solid'}) 

########## HiER xi

        # color_palette = ['orangered','orange','green','blue','navy','purple']
    
        # exps.append({"exp_name": "_".join(['1127_E',alg,'selfpaced','final','predefined','fix','xi01','noper','sparse','Panda'+taskname+'-v3']) , "seed_num":seednum, "color": color_palette[0], "plot_name": "fix 0.1", "linestyle": 'solid'}) # type: ignore
        # exps.append({"exp_name": "_".join(['1127_E',alg,'selfpaced','final','predefined','fix','xi025','noper','sparse','Panda'+taskname+'-v3']) , "seed_num":seednum, "color": color_palette[1], "plot_name": "fix 0.25", "linestyle": 'solid'}) # type: ignore
        # exps.append({"exp_name": "_".join(['1127_E',alg,'selfpaced','final','predefined','fix','xi05','noper','sparse','Panda'+taskname+'-v3']) , "seed_num":seednum, "color": color_palette[2], "plot_name": "fix 0.5", "linestyle": 'solid'}) # type: ignore
        # exps.append({"exp_name": "_".join(['1127_E',alg,'selfpaced','final','predefined','fix','xi075','noper','sparse','Panda'+taskname+'-v3']) , "seed_num":seednum, "color": color_palette[3], "plot_name": "fix 0.75", "linestyle": 'solid'}) # type: ignore
        # exps.append({"exp_name": "_".join(['1127_E',alg,'selfpaced','final','predefined','fix','xi09','noper','sparse','Panda'+taskname+'-v3']) , "seed_num":seednum, "color": color_palette[4], "plot_name": "fix 0.9", "linestyle": 'solid'}) # type: ignore
        # exps.append({"exp_name": "_".join(['1127_E',alg,'selfpaced','final','predefined','prioritized','noper','sparse','Panda'+taskname+'-v3']) , "seed_num":seednum, "color": color_palette[5], "plot_name": "prioritized", "linestyle": 'solid'}) # type: ignore

      
        for plotdata,plotdata_y in zip(plotdata_list,plotdata_y_list):

            exp_test_color_list = []
            #exp_test_linesyte_list = []
            for i in range(len(exps)):                
                if (exps[i]['exp_name'].find('multifix') != -1) and (plotdata in ['hier_batch_size','hier_buffer_size','hier_lambda','hier_xi']):
                    continue
                
                exp_test_color_list.append(exps[i]['color'])
                #exp_test_linesyte_list.append(exps[i]['linestyle'])


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

                    print("-----------------")
                    print(data_pd)

                    data_pd = pd.concat([data_pd,pivot], ignore_index = True) # type: ignore

                        
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
                        #maxdata_pd = pd.concat([maxdata_pd,df_append], ignore_index = True) # type: ignore
                        maxdata_pd.append(df_append, ignore_index = True) # type: ignore
            
                    maxdata_pd['HER'] = False
                    maxdata_pd['PER'] = False
                    maxdata_pd['ISE'] = False
                    maxdata_pd['HiER'] = False

                    for ind in maxdata_pd.index:
                        if maxdata_pd['exp_name'][ind].find('noher') == -1: maxdata_pd['HER'][ind] = True
                        if maxdata_pd['exp_name'][ind].find('noper') == -1: maxdata_pd['PER'][ind] = True
                        if maxdata_pd['exp_name'][ind].find('max') == -1: maxdata_pd['ISE'][ind] = True
                        if maxdata_pd['exp_name'][ind].find('nohier') == -1: maxdata_pd['HiER'][ind] = True
                
                    maxdata_pd = maxdata_pd[['HER','PER','ISE','HiER','meanmaxvalue','stdmaxvalue']]

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
                        plt.plot(np.array(pivot['t']),
                                np.array(pivot['value']),
                                color=exps[i]['color'],
                                label=exps[i]['plot_name'],
                                linestyle = exps[i]['linestyle'],
                                linewidth=linewidth)  
                    else: 
                        plt.plot(np.array(pivot['t']),
                                np.array(pivot['value']),
                                color=exps[i]['color'],
                                linestyle = exps[i]['linestyle'],
                                linewidth=linewidth)
                        

            if args.legend: 
                legend = plt.legend(bbox_to_anchor=(1, 1.01), loc='upper left', fontsize=fontsize_label_next_figure)
                legend.get_frame().set_facecolor('white')
                for line in legend.get_lines():
                    line.set_linewidth(2.0)

            plt.xlabel("Timesteps (x$10^3$)", fontsize=fontsize_axis_label)
            plt.ylabel(plotdata_y, fontsize=fontsize_axis_label)
            plt.title(taskname_title, fontsize=fontsize_title)

        

            x_min = data_pd["t"].min()
            x_max = data_pd["t"].max()+1
            y_min = data_pd["value"].min()
            y_max = data_pd["value"].max()#+0.05 #+5

            #plt.xticks(np.arange(x_min, x_max, step=100),fontsize=fontsize_ticks)
            plt.xticks(fontsize=fontsize_ticks)
            plt.yticks(fontsize=fontsize_ticks)

            plt.xlim([x_min, x_max])
            plt.ylim([y_min, y_max])
            # plt.autoscale(enable=True, axis='x', tight=True)
            # plt.autoscale(enable=True, axis='y', tight=True)

            ax.set_facecolor((1.0, 1.0, 1.0))
            ax.spines['bottom'].set_color('black')
            ax.spines['top'].set_color('black') 
            ax.spines['right'].set_color('black')
            ax.spines['left'].set_color('black')
            plt.grid(c='gray')

            figname = '_'.join([plotid,taskname,plotdata,'.png'])
            plt.savefig(os.path.join(current_dir, args.outdir, logdir, figname),bbox_inches='tight')
            if args.show: plt.show()

            plt.clf()
            plt.cla()

            # SD plot ###################################

            fig, ax = plt.subplots(figsize=(10,8))

            print("#######################xx")
            print(data_pd)
            print(np.array(data_pd))

            sns.lineplot(data=data_pd, 
                         x="t", 
                         y="value", 
                         hue="plot_name", 
                         errorbar=('ci', 95), 
                         palette=exp_test_color_list,
                         legend=args.legend,
                         linewidth = linewidth)

            for i in range(len(exps)): 
                ax.lines[i].set_linestyle(exps[i]['linestyle'])

            if args.legend: 
                legend = plt.legend(bbox_to_anchor=(1, 1.01), loc='upper left', fontsize=fontsize_label_next_figure)
                legend.get_frame().set_facecolor('white')
                for line in legend.get_lines():
                    line.set_linewidth(2.0)
            plt.xlabel("Timesteps")
            plt.ylabel(plotdata_y)
            plt.xlabel("Timesteps (x$10^3$)", fontsize=fontsize_axis_label)
            plt.ylabel(plotdata_y, fontsize=fontsize_axis_label)
            plt.title(taskname_title, fontsize=fontsize_title)
            plt.xticks(fontsize=fontsize_ticks)
            plt.yticks(fontsize=fontsize_ticks)
            plt.xlim([x_min, x_max])
            plt.ylim([y_min, y_max])
            # plt.autoscale(enable=True, axis='x', tight=True)
            # plt.autoscale(enable=True, axis='y', tight=True)
            ax.set_facecolor((1.0, 1.0, 1.0))
            ax.spines['bottom'].set_color('black')
            ax.spines['top'].set_color('black') 
            ax.spines['right'].set_color('black')
            ax.spines['left'].set_color('black')
            plt.grid(c='gray')

            figname = '_'.join([plotid,taskname,plotdata,'std.png'])
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
                    if j == 0:
                        plt.plot(pivot['t'],
                                 pivot['value'],
                                 color=exps[i]['color'],
                                 linestyle = exps[i]['linestyle'],
                                 label=exps[i]['plot_name']) 
                    else: 
                        plt.plot(pivot['t'],
                                 pivot['value'],
                                 color=exps[i]['color'],
                                 linestyle = exps[i]['linestyle'])
                    

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
                figname = '_'.join([plotid,"legend.png"])
                fig.savefig(os.path.join(current_dir, args.outdir, logdir, figname), dpi="figure", bbox_inches=bbox, facecolor='white')

            export_legend(legend)

            plt.clf()
            plt.cla()

