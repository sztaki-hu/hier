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
import scipy

from results_utils import create_folder, generate_latex_table_1, generate_latex_table_2, generate_latex_table_3

sns.set_theme()


# PARAMS ###########################################################
data_name = 'eval_success_rate'  #'eval_success_rate' / 'eval_mean_reward'
og_gammas = [1.0, 1.0, 1.0]
#og_gammas = [-10.0, -20.0, -30.0]
round_digit = 2
results_mode = "best"# last / best


#tasks = ['PandaPush-v3','PandaSlide-v3','PandaPickAndPlace-v3']
#tasks = ['FetchPush-v2','FetchSlide-v2','FetchPickAndPlace-v2']
#tasks = ["PointMaze-Wall-v3","PointMaze-S-v3"]
tasks = ['PandaPush-v3','PandaSlide-v3','PandaPickAndPlace-v3','FetchPush-v2','FetchSlide-v2','FetchPickAndPlace-v2',"PointMaze-Wall-v3","PointMaze-S-v3"]

task_num = len(tasks)
run_num = 10
colors = [(0.5,0.5,0.5),(0.2,0.2,0.2),'green','blue','purple','magenta'] # DUMMY



# OUTPUT ####################################################
output_name = "MAZE_0426_2"
output_dir = os.path.join(current_dir, "results" ,"output", "tables",output_name)

# CREATE FODLER 
create_folder(output_dir)

# INPUT DATA ##############################################################

score_matrix = {}
score_matrix_str = {}

for task, og_gamma in zip(tasks,og_gammas):

    input_data = {}

    algs = []
    pivot = []

    

    # PANDA ALL CONFIG ###############################################x
    # datetag = '1116' if task in ['PandaPush-v3','PandaSlide-v3'] else '1119'
    # for e2h in ['nocl','selfpaced']:
    #     for hier in ['nohier','predefined']:   
    #         for per in ['noper','proportional']:
    #                 for her in ['noher','final']:
    #                     xi_mode = "fix" if per == 'noper' else 'prioritized'
    #                     pivot.append({
    #                         "name":"_".join([her,per,e2h,hier]),
    #                         "file_name": "_".join([datetag,'A','sac',e2h,her,hier,xi_mode,per,'sparse',task])})

    # PANDA SELECTED CONFIG #############################################
    # datetag = '1116' if task in ['PandaPush-v3','PandaSlide-v3'] else '1119'
    # her = 'final'
    # per = 'noper'
    # xi_mode = 'fix'
    # pivot.append({"name":"_".join([her,per,'nocl','nohier']), "file_name": "_".join([datetag,'A','sac','nocl',her,'nohier',xi_mode,per,'sparse',task])})
    # pivot.append({"name":"_".join([her,per,'nocl','predefined']), "file_name": "_".join([datetag,'A','sac','nocl',her,'predefined',xi_mode,per,'sparse',task])})
    # pivot.append({"name":"_".join([her,per,'selfpaced','predefined']), "file_name": "_".join([datetag,'A','sac','selfpaced',her,'predefined',xi_mode,per,'sparse',task])})

    # FETCH ###############################################################
    # letter = 'C' if task == 'FetchPush-v2' else 'A'
   
    # if task != 'FetchSlide-v2':
    #     pivot.append({"name":"_".join(['noher','noper','nocl','nohier']), "file_name": "_".join(['0207','A','sac','noher','nohier','fix','max','noper','sparse',task])}) # type: ignore
    #     pivot.append({"name":"_".join(['final','noper','nocl','nohier']), "file_name": "_".join(['0207','A','sac','final','nohier','fix','max','noper','sparse',task])}) # type: ignore
    #     pivot.append({"name":"_".join(['noher','noper','nocl','predefined']), "file_name": "_".join(['0207','A','sac','noher','predefined','fix','max','noper','sparse',task])}) # type: ignore   
    #     pivot.append({"name":"_".join(['final','noper','nocl','predefined']), "file_name": "_".join(['0207',letter,'sac','final','predefined','fix','max','noper','sparse',task])}) # type: ignore
    # else:
    #     pivot.append({"name":"_".join(['noher','noper','nocl','nohier']), "file_name": "_".join(['0207','A','sac','noher','nohier','fix','max','noper','sparse',task])}) # type: ignore
    #     pivot.append({"name":"_".join(['final','noper','nocl','nohier']), "file_name": "_".join(['0211','F','sac','final_valid','nohier','fix','max','noper','sparse',task])}) # type: ignore
    #     pivot.append({"name":"_".join(['noher','noper','nocl','predefined']), "file_name": "_".join(['0207','A','sac','noher','predefined','fix','max','noper','sparse',task])}) # type: ignore
    #     pivot.append({"name":"_".join(['final','noper','nocl','predefined']), "file_name": "_".join(['0211','F','sac','final_valid','predefined','fix','max','noper','sparse',task])}) # type: ignore

    # POINTMAZE ###############################################################
    
    # ##### "PointMaze-Wall-v3"
    # if task == "PointMaze-Wall-v3":
    #     pivot.append({"name":"_".join(['noher','noper','nocl','nohier']),"file_name": "_".join(['0207','B','sac','noher','nohier','fix','max',"7x7_wall",'noper','sparse','PointMaze_UMaze-v3'])}) # type: ignore
    #     pivot.append({"name":"_".join(['final','noper','nocl','nohier']),"file_name": "_".join(['0212','B','sac','final','nohier','fix','max',"7x7_wall",'noper','sparse','PointMaze_UMaze-v3'])}) # type: ignore
    #     pivot.append({"name":"_".join(['noher','noper','nocl','predefined']),"file_name": "_".join(['0207','B','sac','noher','predefined','fix','max',"7x7_wall",'noper','sparse','PointMaze_UMaze-v3'])}) # type: ignore    
    #     pivot.append({"name":"_".join(['final','noper','nocl','predefined']),"file_name": "_".join(['0212','B','sac','final','predefined','fix','max',"7x7_wall",'noper','sparse','PointMaze_UMaze-v3'])}) # type: ignore      
        
    # #### "PointMaze-S-v3"
    # if task == "PointMaze-S-v3":
    #     pivot.append({"name":"_".join(['noher','noper','nocl','nohier']),"file_name": "_".join(['0211','B','sac','gam10','alp01','noher','nohier','fix','max',"7x7_S",'noper','sparse','PointMaze_UMaze-v3'])}) # type: ignore
    #     # exps.append({"exp_name": "_".join(['0209','B',alg,'gam10','alp01','noher','nohier','fix','max',"7x7_S",'noper','sparse',taskname]) , "seed_num":seednum, "color": color_palette[0], "plot_name": "Baseline" , "linestyle": 'dashed'}) # type: ignore
    #     pivot.append({"name":"_".join(['final','noper','nocl','nohier']),"file_name": "_".join(['0209','B','sac','gam10','alp01','final','nohier','fix','max',"7x7_S",'noper','sparse','PointMaze_UMaze-v3'])}) # type: ignore
    #     pivot.append({"name":"_".join(['noher','noper','nocl','predefined']),"file_name": "_".join(['0209','B','sac','gam10','alp01','noher','predefined','fix','max',"7x7_S",'noper','sparse','PointMaze_UMaze-v3'])}) # type: ignore  
    #     pivot.append({"name":"_".join(['final','noper','nocl','predefined']),"file_name": "_".join(['0209','B','sac','gam10','alp01','final','predefined','fix','max',"7x7_S",'noper','sparse','PointMaze_UMaze-v3'])}) # type: ignore      

    # AGGREGATED SCORE

    if task in ['PandaPush-v3','PandaSlide-v3','PandaPickAndPlace-v3']:
        datetag = '1116' if task in ['PandaPush-v3','PandaSlide-v3'] else '1119'
        pivot.append({"name": "_".join(['noher','noper','nocl','nohier']), "file_name": "_".join([datetag,'A','sac','nocl','noher','nohier','fix','noper','sparse',task]), "color": colors[0], "linestyle": 'dashed'})       
        pivot.append({"name": "_".join(['final','noper','nocl','nohier']), "file_name": "_".join([datetag,'A','sac','nocl','final','nohier','fix','noper','sparse',task]) ,"color": colors[1], "linestyle": 'dashed' })  
        pivot.append({"name": "_".join(['noher','noper','nocl','predefined']), "file_name": "_".join([datetag,'A','sac','nocl','noher','predefined','fix','noper','sparse',task]) ,"color": colors[3], "linestyle": 'solid' })       
        pivot.append({"name": "_".join(['final','noper','nocl','predefined']), "file_name": "_".join([datetag,'A','sac','nocl','final','predefined','fix','noper','sparse',task]) ,"color": colors[3], "linestyle": 'solid' }) 
   
    if task in ['FetchPush-v2','FetchSlide-v2','FetchPickAndPlace-v2']:
        letter = 'C' if task == 'FetchPush-v2' else 'A'
        if task != 'FetchSlide-v2':
            pivot.append({"name": "_".join(['noher','noper','nocl','nohier']), "file_name": "_".join(['0207','A','sac','noher','nohier','fix','max','noper','sparse',task]), "color": colors[0], "linestyle": 'dashed'}) # type: ignore
            pivot.append({"name": "_".join(['final','noper','nocl','nohier']), "file_name": "_".join(['0207','A','sac','final','nohier','fix','max','noper','sparse',task]), "color": colors[2], "linestyle": 'dashed'}) # type: ignore
            pivot.append({"name": "_".join(['noher','noper','nocl','predefined']), "file_name": "_".join(['0207','A','sac','noher','predefined','fix','max','noper','sparse',task]), "color": colors[1], "linestyle": 'solid'}) # type: ignore   
            pivot.append({"name": "_".join(['final','noper','nocl','predefined']), "file_name": "_".join(['0207',letter,'sac','final','predefined','fix','max','noper','sparse',task]), "color": colors[3], "linestyle": 'solid'}) # type: ignore
        else:
            pivot.append({"name": "_".join(['noher','noper','nocl','nohier']), "file_name": "_".join(['0207','A','sac','noher','nohier','fix','max','noper','sparse',task]), "color": colors[0], "linestyle": 'dashed'}) # type: ignore           
            pivot.append({"name": "_".join(['final','noper','nocl','nohier']), "file_name": "_".join(['0211','F','sac','final_valid','nohier','fix','max','noper','sparse',task]), "color": colors[2], "linestyle": 'dashed'}) # type: ignore  
            pivot.append({"name": "_".join(['noher','noper','nocl','predefined']), "file_name": "_".join(['0207','A','sac','noher','predefined','fix','max','noper','sparse',task]), "color": colors[1], "linestyle": 'solid'}) # type: ignore
            pivot.append({"name": "_".join(['final','noper','nocl','predefined']), "file_name": "_".join(['0211','F','sac','final_valid','predefined','fix','max','noper','sparse',task]), "color": colors[3], "linestyle": 'solid'}) # type: ignore

    if task in ["PointMaze-Wall-v3","PointMaze-S-v3"]:
          ##### "PointMaze-Wall-v3"
        if task == "PointMaze-Wall-v3":
            pivot.append({"name":"_".join(['noher','noper','nocl','nohier']),"file_name": "_".join(['0207','B','sac','noher','nohier','fix','max',"7x7_wall",'noper','sparse','PointMaze_UMaze-v3'])}) # type: ignore      
            pivot.append({"name":"_".join(['final','noper','nocl','nohier']),"file_name": "_".join(['0212','B','sac','final','nohier','fix','max',"7x7_wall",'noper','sparse','PointMaze_UMaze-v3'])}) # type: ignore     
            pivot.append({"name":"_".join(['noher','noper','nocl','predefined']),"file_name": "_".join(['0207','B','sac','noher','predefined','fix','max',"7x7_wall",'noper','sparse','PointMaze_UMaze-v3'])}) # type: ignore    
            pivot.append({"name":"_".join(['final','noper','nocl','predefined']),"file_name": "_".join(['0212','B','sac','final','predefined','fix','max',"7x7_wall",'noper','sparse','PointMaze_UMaze-v3'])}) # type: ignore      
            
        #### "PointMaze-S-v3"
        if task == "PointMaze-S-v3":
            pivot.append({"name":"_".join(['noher','noper','nocl','nohier']),"file_name": "_".join(['0211','B','sac','gam10','alp01','noher','nohier','fix','max',"7x7_S",'noper','sparse','PointMaze_UMaze-v3'])}) # type: ignore     
            pivot.append({"name":"_".join(['final','noper','nocl','nohier']),"file_name": "_".join(['0209','B','sac','gam10','alp01','final','nohier','fix','max',"7x7_S",'noper','sparse','PointMaze_UMaze-v3'])}) # type: ignore    
            pivot.append({"name":"_".join(['noher','noper','nocl','predefined']),"file_name": "_".join(['0209','B','sac','gam10','alp01','noher','predefined','fix','max',"7x7_S",'noper','sparse','PointMaze_UMaze-v3'])}) # type: ignore  
            pivot.append({"name":"_".join(['final','noper','nocl','predefined']),"file_name": "_".join(['0209','B','sac','gam10','alp01','final','predefined','fix','max',"7x7_S",'noper','sparse','PointMaze_UMaze-v3'])}) # type: ignore      
            
     

    input_data[task] = pivot

    for alg in pivot:
        algs.append(alg["name"])

    alg_num = len(algs)

    data = {}

    for alg_i in range(alg_num):

        alg_data_matrix = np.zeros((run_num, 1))

        input_data_task = input_data[task]

        for run_i in range(run_num):
            
            path = os.path.join(current_dir, "logs",input_data_task[alg_i]['file_name'],str(run_i),'runs','csv',data_name+'.csv')
            pivot = pd.read_csv(path)

            column_names = list(pivot.columns)
            pivot = pivot.rename(columns={column_names[0]: 't', column_names[1]: "value"})

            if results_mode == 'last':
                value_store = pivot.iloc[-1]['value']
            elif results_mode == 'best':
                value_store = pivot['value'].max()

            alg_data_matrix[run_i][0] = value_store

        data[input_data_task[alg_i]['name']] = alg_data_matrix


    score_matrix[task] = np.zeros((alg_num, 6))

    for alg_i in range(alg_num):
        scores = data[algs[alg_i]]

        score_matrix[task][alg_i][0] = round(np.mean(scores),round_digit) # MEAN
        score_matrix[task][alg_i][1]  = round(np.median(scores),round_digit) # MEDIAN
        score_matrix[task][alg_i][2]  = round(scipy.stats.trim_mean(scores, proportiontocut=0.25, axis=None),round_digit) # IQM
        score_matrix[task][alg_i][3]  = round(og_gamma - np.mean(np.minimum(scores, og_gamma)),round_digit) # Optimality Gap
        score_matrix[task][alg_i][4]  = round(np.max(scores),round_digit) # MAX
        score_matrix[task][alg_i][5]  = round(np.std(scores),round_digit) # STD


    max_values = np.max(score_matrix[task], axis=0)
    min_values = np.min(score_matrix[task], axis=0)
    mask = [True, True, True, False, True, False]
    bold_vals = [x if mask[i] else y for i, (x, y) in enumerate(zip(max_values, min_values))]

    score_matrix_str[task] = np.empty((alg_num, 7), dtype=object)

    for alg_i in range(alg_num):
        
        # score_matrix_str[alg_i][0] = "\checkmark" if algs[alg_i].find('noher') == -1 else "-"
        # score_matrix_str[alg_i][1] = "\checkmark" if algs[alg_i].find('noper') == -1 else "-"
        # score_matrix_str[alg_i][2] = "\checkmark" if algs[alg_i].find('nocl') == -1 else "-"
        # score_matrix_str[alg_i][3] = "\checkmark" if algs[alg_i].find('nohier') == -1 else "-"

        # alg_string = ""
        # alg_string += " \checkmark " if algs[alg_i].find('noher') == -1 else " - "
        # alg_string += " \checkmark " if algs[alg_i].find('noper') == -1 else " - "
        # alg_string += " \checkmark " if algs[alg_i].find('nocl') == -1 else " -"
        # alg_string += " \checkmark " if algs[alg_i].find('nohier') == -1 else " - "

        score_matrix_str[task][alg_i][0] = ""

        for metrics_i in range(6):
            score = score_matrix[task][alg_i][metrics_i] 
            if round_digit == 1:
                if abs(score) < 10:
                    score_matrix_str[task][alg_i][metrics_i+1] = "\\hspace{1.25pt} \\textbf{%.1f}" % score if score == bold_vals[metrics_i] else "\\hspace{1.25pt} %.1f" % score
                else:
                     score_matrix_str[task][alg_i][metrics_i+1] = "\\textbf{%.1f}" % score if score == bold_vals[metrics_i] else "%.1f" % score
            elif round_digit == 2:
                score_matrix_str[task][alg_i][metrics_i+1] = "\\textbf{%.2f}" % score if score == bold_vals[metrics_i] else "%.2f" % score


        
combined_score_matrix_str = np.empty((alg_num, 7), dtype=object)
for alg_i in range(alg_num):
    
    alg_string = ""
    alg_string += " \checkmark " if algs[alg_i].find('noher') == -1 else " - "
    alg_string += " \checkmark " if algs[alg_i].find('noper') == -1 else " - "
    alg_string += " \checkmark " if algs[alg_i].find('nocl') == -1 else " -"
    alg_string += " \checkmark " if algs[alg_i].find('nohier') == -1 else " - "

    combined_score_matrix_str[alg_i][0] = alg_string
    for i in range(1,7):
        if task_num == 3:
            combined_score_matrix_str[alg_i][i] = " | ".join([score_matrix_str[tasks[0]][alg_i][i],score_matrix_str[tasks[1]][alg_i][i],score_matrix_str[tasks[2]][alg_i][i]]) 
        elif task_num == 2:
            combined_score_matrix_str[alg_i][i] = " | ".join([score_matrix_str[tasks[0]][alg_i][i],score_matrix_str[tasks[1]][alg_i][i]]) 
        elif task_num == 1:
            combined_score_matrix_str[alg_i][i] = score_matrix_str[tasks[0]][alg_i][i]
        else:
            combined_score_matrix_str[alg_i][i]

if task_num == 3:
    latex_code = generate_latex_table_3(combined_score_matrix_str, tasks)
elif task_num == 2:
    latex_code = generate_latex_table_2(combined_score_matrix_str, tasks)
elif task_num == 1:
    latex_code = generate_latex_table_1(combined_score_matrix_str, tasks)

with open(os.path.join(output_dir, output_name+"_"+results_mode+"_"+data_name+".txt"), "w") as text_file:
    text_file.write(latex_code)

   


