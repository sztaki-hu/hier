import os
from os.path import dirname, abspath
current_dir = dirname(abspath(__file__))
current_dir = os.path.abspath(os.path.join(current_dir, os.pardir))

import numpy as np
import pandas as pd
import argparse

from rliable import library as rly
from rliable import metrics
from rliable import plot_utils

# Import seaborn
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()

from results_utils import create_folder

# PARAMS ###########################################################
#tasks = ['PandaSlide-v3']
#tasks = ['PandaPush-v3','PandaSlide-v3','PandaPickAndPlace-v3']
tasks = ['FetchPush-v2','FetchSlide-v2','FetchPickAndPlace-v2']
#tasks = ['PandaPush-v3','PandaSlide-v3','PandaPickAndPlace-v3','FetchPush-v2','FetchSlide-v2','FetchPickAndPlace-v2',"PointMaze-Wall-v3","PointMaze-S-v3"]
task_num = len(tasks)
run_num = 10

data_names = ['eval_success_rate'] #'eval_success_rate' / 'eval_mean_reward'
data_plot_names = ['Success rate score'] #'Success rate score' / 'Reward'
results_mode = "best"# last / best


# CONTROL COMPUTATION #################################################
reps_scalor = 1.0

# CONTROL PLOTTING ####################################################
output_name = "Fetch_0429_success"
output_dir = os.path.join(current_dir, "results" ,"output", "plot_stat",output_name)

plot_agg_metrics = True
plot_prob = False
plot_perf_profiles = False
plot_hist = False


# CREATE FODLER 
create_folder(output_dir)

# INPUT DATA ##############################################################

input_data = {}

algs = []
linestyles = []

for task in tasks:

    pivot = []

    # PANDA #######################################################################

    # colors = [(0.5,0.5,0.5),(0.2,0.2,0.2),'green','blue','purple','magenta']
    # datetag = '1116' if task in ['PandaPush-v3','PandaSlide-v3'] else '1119'
    # xlabel_y_coordinate=-0.25
            
    # pivot.append({"name": "Baseline", "file_name": "_".join([datetag,'A','sac','nocl','noher','nohier','fix','noper','sparse',task]), "color": colors[0], "linestyle": 'dashed'})
    # pivot.append({"name": "Baseline [HER & PER]", "file_name": "_".join([datetag,'A','sac','nocl','final','nohier','prioritized','proportional','sparse',task]) ,"color": colors[1], "linestyle": 'dashed' })        
    # pivot.append({"name": "E2H-ISE [HER & PER]","file_name": "_".join([datetag,'A','sac','selfpaced','final','nohier','prioritized','proportional','sparse',task]) , "color": colors[2], "linestyle": 'solid' }) 
    # pivot.append({"name": "HiER [HER & PER]", "file_name": "_".join([datetag,'A','sac','nocl','final','predefined','prioritized','proportional','sparse',task]) ,"color": colors[3], "linestyle": 'solid' }) 
    # pivot.append({"name": "HiER+ [HER & PER]","file_name": "_".join([datetag,'A','sac','selfpaced','final','predefined','prioritized','proportional','sparse',task]) , "color": colors[4], "linestyle": 'solid' }) 
    # pivot.append({"name": "HiER+ [HER]","file_name": "_".join([datetag,'A','sac','selfpaced','final','predefined','fix','noper','sparse',task]) , "color": colors[5], "linestyle": 'solid' }) 
    

    # FETCH ###############################################################################

    colors = [(0.0,0.4,0.6),(0.0,0.0,1.0),'purple','magenta']

    letter = 'C' if task == 'FetchPush-v2' else 'A'
    if task != 'FetchSlide-v2':
        pivot.append({"name": "Baseline", "file_name": "_".join(['0207','A','sac','noher','nohier','fix','max','noper','sparse',task]), "color": colors[0], "linestyle": 'dashed'}) # type: ignore
        pivot.append({"name": "HiER", "file_name": "_".join(['0207','A','sac','noher','predefined','fix','max','noper','sparse',task]), "color": colors[1], "linestyle": 'solid'}) # type: ignore   
        pivot.append({"name": "Baseline [HER]", "file_name": "_".join(['0207','A','sac','final','nohier','fix','max','noper','sparse',task]), "color": colors[2], "linestyle": 'dashed'}) # type: ignore
        pivot.append({"name": "HiER [HER]", "file_name": "_".join(['0207',letter,'sac','final','predefined','fix','max','noper','sparse',task]), "color": colors[3], "linestyle": 'solid'}) # type: ignore
    else:
        pivot.append({"name": "Baseline", "file_name": "_".join(['0207','A','sac','noher','nohier','fix','max','noper','sparse',task]), "color": colors[0], "linestyle": 'dashed'}) # type: ignore
        pivot.append({"name": "HiER", "file_name": "_".join(['0207','A','sac','noher','predefined','fix','max','noper','sparse',task]), "color": colors[1], "linestyle": 'solid'}) # type: ignore
        pivot.append({"name": "Baseline [HER]", "file_name": "_".join(['0211','F','sac','final_valid','nohier','fix','max','noper','sparse',task]), "color": colors[2], "linestyle": 'dashed'}) # type: ignore  
        pivot.append({"name": "HiER [HER]", "file_name": "_".join(['0211','F','sac','final_valid','predefined','fix','max','noper','sparse',task]), "color": colors[3], "linestyle": 'solid'}) # type: ignore

    xlabel_y_coordinate=-0.5


    # AGGREGATE ###############################################################################

    # colors = [(0.0,0.4,0.6),'blue','purple','magenta']
    # xlabel_y_coordinate=-0.5
    
    # if task in ['PandaPush-v3','PandaSlide-v3','PandaPickAndPlace-v3']:
    #     datetag = '1116' if task in ['PandaPush-v3','PandaSlide-v3'] else '1119'
    #     pivot.append({"name": "Baseline", "file_name": "_".join([datetag,'A','sac','nocl','noher','nohier','fix','noper','sparse',task]), "color": colors[0], "linestyle": 'dashed'})        
    #     pivot.append({"name": "HiER", "file_name": "_".join([datetag,'A','sac','nocl','noher','predefined','fix','noper','sparse',task]) ,"color": colors[3], "linestyle": 'solid' })       
    #     pivot.append({"name": "Baseline [HER]", "file_name": "_".join([datetag,'A','sac','nocl','final','nohier','fix','noper','sparse',task]) ,"color": colors[1], "linestyle": 'dashed' })  
    #     pivot.append({"name": "HiER [HER]", "file_name": "_".join([datetag,'A','sac','nocl','final','predefined','fix','noper','sparse',task]) ,"color": colors[3], "linestyle": 'solid' }) 
   
    # if task in ['FetchPush-v2','FetchSlide-v2','FetchPickAndPlace-v2']:
    #     letter = 'C' if task == 'FetchPush-v2' else 'A'
    #     if task != 'FetchSlide-v2':
    #         pivot.append({"name": "Baseline", "file_name": "_".join(['0207','A','sac','noher','nohier','fix','max','noper','sparse',task]), "color": colors[0], "linestyle": 'dashed'}) # type: ignore    
    #         pivot.append({"name": "HiER", "file_name": "_".join(['0207','A','sac','noher','predefined','fix','max','noper','sparse',task]), "color": colors[1], "linestyle": 'solid'}) # type: ignore   
    #         pivot.append({"name": "Baseline [HER]", "file_name": "_".join(['0207','A','sac','final','nohier','fix','max','noper','sparse',task]), "color": colors[2], "linestyle": 'dashed'}) # type: ignore
    #         pivot.append({"name": "HiER [HER]", "file_name": "_".join(['0207',letter,'sac','final','predefined','fix','max','noper','sparse',task]), "color": colors[3], "linestyle": 'solid'}) # type: ignore
    #     else:
    #         pivot.append({"name": "Baseline", "file_name": "_".join(['0207','A','sac','noher','nohier','fix','max','noper','sparse',task]), "color": colors[0], "linestyle": 'dashed'}) # type: ignore               
    #         pivot.append({"name": "HiER", "file_name": "_".join(['0207','A','sac','noher','predefined','fix','max','noper','sparse',task]), "color": colors[1], "linestyle": 'solid'}) # type: ignore
    #         pivot.append({"name": "Baseline [HER]", "file_name": "_".join(['0211','F','sac','final_valid','nohier','fix','max','noper','sparse',task]), "color": colors[2], "linestyle": 'dashed'}) # type: ignore  
    #         pivot.append({"name": "HiER [HER]", "file_name": "_".join(['0211','F','sac','final_valid','predefined','fix','max','noper','sparse',task]), "color": colors[3], "linestyle": 'solid'}) # type: ignore

    # if task in ["PointMaze-Wall-v3","PointMaze-S-v3"]:
    #       ##### "PointMaze-Wall-v3"
    #     if task == "PointMaze-Wall-v3":
    #         pivot.append({"name":"Baseline","file_name": "_".join(['0207','B','sac','noher','nohier','fix','max',"7x7_wall",'noper','sparse','PointMaze_UMaze-v3']), "linestyle": 'dashed'}) # type: ignore         
    #         pivot.append({"name":"HiER","file_name": "_".join(['0207','B','sac','noher','predefined','fix','max',"7x7_wall",'noper','sparse','PointMaze_UMaze-v3']),"linestyle": 'solid'}) # type: ignore    
    #         pivot.append({"name":"Baseline [HER]","file_name": "_".join(['0212','B','sac','final','nohier','fix','max',"7x7_wall",'noper','sparse','PointMaze_UMaze-v3']), "linestyle": 'dashed'}) # type: ignore     
    #         pivot.append({"name":"HiER [HER]","file_name": "_".join(['0212','B','sac','final','predefined','fix','max',"7x7_wall",'noper','sparse','PointMaze_UMaze-v3']), "linestyle": 'solid'}) # type: ignore      
            
    #     #### "PointMaze-S-v3"
    #     if task == "PointMaze-S-v3":
    #         pivot.append({"name":"Baseline","file_name": "_".join(['0211','B','sac','gam10','alp01','noher','nohier','fix','max',"7x7_S",'noper','sparse','PointMaze_UMaze-v3']), "linestyle": 'dashed'}) # type: ignore            
    #         pivot.append({"name":"HiER","file_name": "_".join(['0209','B','sac','gam10','alp01','noher','predefined','fix','max',"7x7_S",'noper','sparse','PointMaze_UMaze-v3']), "linestyle": 'solid'}) # type: ignore  
    #         pivot.append({"name":"Baseline [HER]","file_name": "_".join(['0209','B','sac','gam10','alp01','final','nohier','fix','max',"7x7_S",'noper','sparse','PointMaze_UMaze-v3']), "linestyle": 'dashed'}) # type: ignore    
    #         pivot.append({"name":"HiER [HER]","file_name": "_".join(['0209','B','sac','gam10','alp01','final','predefined','fix','max',"7x7_S",'noper','sparse','PointMaze_UMaze-v3']), "linestyle": 'solid'}) # type: ignore      
            
    

    input_data[task] = pivot

for alg in pivot:
    algs.append(alg["name"])
    linestyles.append(alg["linestyle"])

print(algs)
alg_num = len(algs)
print(linestyles)



#print(exps)

for data_name,data_plot_name in zip(data_names,data_plot_names):

    data = {}

    for alg_i in range(alg_num):

        alg_data_matrix = np.zeros((run_num, task_num))

        for task_i in range(task_num):

            input_data_task = input_data[tasks[task_i]]

            for run_i in range(run_num):
                
                path = os.path.join(current_dir, "logs",input_data_task[alg_i]['file_name'],str(run_i),'runs','csv',data_name+'.csv')
                #print(path)
                pivot = pd.read_csv(path)

                column_names = list(pivot.columns)
                pivot = pivot.rename(columns={column_names[0]: 't', column_names[1]: "value"})
                #print(pivot)

                if results_mode == 'last':
                    value_store = pivot.iloc[-1]['value']
                elif results_mode == 'best':
                    value_store = pivot['value'].max()

                alg_data_matrix[run_i][task_i] = value_store


        data[input_data_task[alg_i]['name']] = alg_data_matrix
    

#print(data)
        
###########################################################################
## Agragate Metrics #######################################################
###########################################################################
if plot_agg_metrics:
    aggregate_func = lambda x: np.array([
        metrics.aggregate_mean(x),
        metrics.aggregate_median(x),
        metrics.aggregate_iqm(x),     
        metrics.aggregate_optimality_gap(x)])

    aggregate_scores, aggregate_score_cis = rly.get_interval_estimates(
        data, aggregate_func, reps=int(5000*reps_scalor))
    
    print(aggregate_scores)

    fig, axes = plot_utils.plot_interval_estimates(
        aggregate_scores, aggregate_score_cis,
        metric_names=['Mean','Median', 'IQM', 'Optimality Gap'],
        algorithms=algs, 
        xlabel=data_plot_name,
        colors=dict(zip(algs, colors)),
        max_ticks=3,
        subfigure_width=3.4, #3.4
        xlabel_y_coordinate=xlabel_y_coordinate)


    for ax in axes:
        ax.set_facecolor((1.0, 1.0, 1.0))
        ax.grid(axis='x',c='gray')
        ax.spines['bottom'].set_color('black')

    #fig.set_size_inches(20, 6)

    plt.savefig(os.path.join(output_dir,"_".join([output_name,"aggregate_metrics",data_name,results_mode,".pdf"])),bbox_inches='tight')
    
    
    plt.clf()
    plt.cla()

###########################################################################
## Probability ############################################################
###########################################################################

if plot_prob:   
    pairs = []

    # Iterate over each element in the list
    for i in range(len(algs)):
        for j in range(i+1, len(algs)):
            # Append a tuple of the pair to the pairs list
            pairs.append((algs[i], algs[j])) 

    # Remove some pairs  
    #pairs.pop(0)
    
    pairs_subsets = []
    #pairs_subsets.append(pairs[:8] + pairs[11:])
    #pairs_subsets.append(pairs[11:])
    #pairs_subsets.append(pairs[4:8])
    #pairs_subsets.append(pairs[8:])

    pairs = [("Baseline","HiER"),("Baseline [HER]","HiER [HER]"),("HiER","HiER [HER]")]
    pairs_subsets.append(pairs)


    #colors_prob = ['green','blue','purple','magenta','green','blue','purple','magenta','purple','magenta','magenta']
    #colors_prob = ['purple','magenta','magenta']
    colors_prob = ['blue','magenta','magenta']

    prob_index = 0
    for pairs_subset in pairs_subsets:
        algorithm_pairs = {}
        for pair in pairs_subset :
            algorithm_pairs[pair[1]+","+pair[0]] = (data[pair[1]],data[pair[0]])

        #print(algorithm_pairs)

        average_probabilities, average_prob_cis = rly.get_interval_estimates(
        algorithm_pairs, metrics.probability_of_improvement, reps=int(2000*reps_scalor))

        print("--------------------------")
        print("AVG probabilities")
        print(average_probabilities)

        ax = plot_utils.plot_probability_of_improvement(
            average_probabilities, 
            average_prob_cis,
            colors=colors_prob)

        ax.set_facecolor((1.0, 1.0, 1.0))
        ax.grid(axis='x',c='gray')
        ax.spines['bottom'].set_position(('data', -0.65))  # Set position to zero
        ax.spines['bottom'].set_color('black') 
  

        plt.savefig(os.path.join(output_dir, "_".join([output_name,"probability",str(prob_index),data_name,results_mode,".pdf"])),bbox_inches='tight')
        
        plt.clf()
        plt.cla()
        prob_index+=1


###########################################################################
## Performance Profiles ###################################################
###########################################################################

if plot_perf_profiles:
    thresholds = np.linspace(0.0, 1.0, 81)
    score_distributions, score_distributions_cis = rly.create_performance_profile(
        data, thresholds)
    # Plot score distributions
    fig, ax = plt.subplots(ncols=1, figsize=(7, 5))
    plot_utils.plot_performance_profiles(
        score_distributions, thresholds,
        performance_profile_cis=score_distributions_cis,
        colors=dict(zip(algs, colors)),
        xlabel= data_plot_name + r'$ (\tau)$',
        linestyles=linestyles,
        ax=ax)

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),frameon=False, ncol=2, facecolor='white',fontsize=18)
    
    # handles, labels = plt.gca().get_legend_handles_labels()
    # handles = handles[::-1]
    # labels = labels[::-1]   
    # plt.legend(handles, labels,loc='center left', bbox_to_anchor=(1, 0.5), frameon=False, ncol=1, facecolor='white', fontsize=16, title_fontsize=-16)


    plt.axhline(y=0.5, color='red', linestyle="dotted", label='y=0.5')

    plt.xlim(0, 1.05)
    plt.ylim(0, 1.05)
    ax.set_facecolor((1.0, 1.0, 1.0))
    ax.grid(axis='both',c='gray')
    ax.spines['bottom'].set_position(('data', 0))  # Set position to zero
    ax.spines['left'].set_position(('data', 0))  # Set position to zero
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.tick_params(axis='x', which='both', bottom=True, labelbottom=True, pad=8)

    plt.savefig(os.path.join(output_dir, "_".join([output_name,"performance_profile",data_name,results_mode,".pdf"])),bbox_inches='tight')
    plt.clf()
    plt.cla()


###########################################################################
## Plot Histogram #########################################################
###########################################################################
# if plot_hist:
#     create_folder(os.path.join(output_dir,"hist"))
#     for alg in algs:
#         for task_id in range(task_num):

#             data_alg = data[alg][:,task_id] 

#             # Plot the histogram
#             plt.hist(data_alg, bins=30, color='cornflowerblue', alpha=0.7, density=True)

#             # Overlay a kernel density estimate (KDE) plot
#             sns.kdeplot(data_alg, color='blue', linestyle='-', linewidth=2)
            
#             plt.xlim(0, 1)
#             plt.xlabel('Value')
#             plt.ylabel('Density')
#             plt.title('Histogram with KDE')
#             plt.grid(True)

#             plt.savefig(os.path.join(output_dir, "hist",output_name+"_histograms_"+alg+"_"+tasks[task_id]+".png"),bbox_inches='tight')
#             plt.clf()
#             plt.cla()

print("Done")
