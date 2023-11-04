import os
from os.path import dirname, abspath
current_dir = dirname(abspath(__file__))

import numpy as np
import pandas as pd
import argparse

# Import seaborn
import matplotlib.pyplot as plt
import matplotlib
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
parser.add_argument("--show", default=False ,help="-")
parser.add_argument("--createcsv", default=True ,help="-")
parser.add_argument("--legend", default=False ,help="-")
args = parser.parse_args()

fontsize_title = 32
fontsize_label = 28
fontsize_label_next_figure = 18
fontsize_ticks = 22

fontsize_axis_label = 28
legend_columns = 6
color_line_width = 4.0

plotid = 'X_1104_CL_3'
seednum = 2
#plotdata_list = ['eval_success_rate',"rollout_success_rate","cl_ratio","rollout_state_changed","eval_state_change_rate","hl_highlights_buffer_size","time_fps"]
#taskname_list = ['Reach','Push','Slide','PickAndPlace','Stack']
#taskname_list = ['Push','Slide','PickAndPlace']
#taskname_list = ['Push','Slide']
taskname_list = ['Push']
#taskname_list = ['Hopper','Walker2d','Ant']
#taskname_list = ['Hopper','Walker2d','HalfCheetah','Ant']
#taskname_list = ['PickAndPlace']
plotdata_list = ['eval_success_rate','eval_mean_reward',
                 'rollout_success_rate', 'rollout_state_changed','rollout_ep_rew_mean',
                 'cl_ratio',
                 'time_fps',
                 'hl_highlights_threshold','hl_highlights_batch_ratio','hl_highlights_buffer_size']
#plotdata_list = ['hl_highlights_threshold','hl_highlights_batch_ratio','hl_highlights_buffer_size']

# plotdata_list = ['eval_success_rate','eval_mean_reward']
# plotdata_y_list = ['Eval success rate','Eval sum reward']

plotdata_list = ['eval_success_rate']
plotdata_y_list = ['Eval success rate']

create_folder(os.path.join(current_dir, args.outdir))

algs = ['sac']


for taskname in taskname_list:
    for alg in algs:
    
        logdir = plotid + "_Panda" + taskname + "_" + alg
        create_folder(os.path.join(current_dir, args.outdir, logdir))
        exps = []

        #exps.append({"exp_name": "_".join(['1012_B', 'Panda'+taskname+'-v3',alg,'sparse','noher','nohl','noper','nocl']) , "seed_num":seednum, "color": "brown", "plot_name": alg})  
        # exps.append({"exp_name": "_".join(['1012_B', 'Panda'+taskname+'-v3',alg,'sparse','final','nohl','noper','nocl']) , "seed_num":seednum, "color": "orange", "plot_name": alg + " HER"}) 

        # exps.append({"exp_name": "_".join(['1012_B', 'Panda'+taskname+'-v3',alg,'sparse','noher','nohl','proportional','nocl']) , "seed_num":seednum, "color": "darkkhaki", "plot_name":  alg +" PER"}) #ddpg
        # exps.append({"exp_name": "_".join(['1012_B', 'Panda'+taskname+'-v3',alg,'sparse','final','nohl','proportional','nocl']) , "seed_num":seednum, "color": "gold", "plot_name":  alg +" HER+PER"}) #td3
        
        # exps.append({"exp_name": "_".join(['1012_B', 'Panda'+taskname+'-v3',alg,'sparse','final','fix','proportional','controldiscreteadaptive']) , "seed_num":seednum, "color": "blue", "plot_name":  alg +" HER+PER+HiER(fix)+CL"})
        # exps.append({"exp_name": "_".join(['1012_B', 'Panda'+taskname+'-v3',alg,'sparse','final','ama','proportional','controldiscreteadaptive']) , "seed_num":seednum, "color": "navy", "plot_name":  alg +" HER+PER+HiER(ama)+CL"})
        # exps.append({"exp_name": "_".join(['1015_A', 'Panda'+taskname+'-v3',alg,'sparse','noher','ama','noper','nocl']) , "seed_num":seednum, "color": "lightblue", "plot_name":  alg +" HiER"})
        # exps.append({"exp_name": "_".join(['1015_A', 'Panda'+taskname+'-v3',alg,'sparse','noher','nohl','noper','controldiscreteadaptive']) , "seed_num":seednum, "color": "orange", "plot_name":  alg +" CL"})
        # exps.append({"exp_name": "_".join(['1015_A', 'Panda'+taskname+'-v3',alg,'state_change_bonus','final','ama','proportional','controldiscreteadaptive']) , "seed_num":seednum, "color": "purple", "plot_name":  alg +" HER+PER+HiER(fix)+CL+REW"})

        # exps.append({"exp_name": "_".join(['1017_A', 'Panda'+taskname+'-v3',alg,'sparse','final','ama','fix','noper','controldiscreteadaptive']) , "seed_num":seednum, "color": "lime", "plot_name":  alg +" HER+HiER(ama)+CL"})
        # exps.append({"exp_name": "_".join(['1017_A', 'Panda'+taskname+'-v3',alg,'sparse','final','ama','prioritized','noper','controldiscreteadaptive']) , "seed_num":seednum, "color": "aqua", "plot_name":  alg +" HER+PHiER(ama)+CL"})
        
        # exps.append({"exp_name": "_".join(['1017_A', 'Panda'+taskname+'-v3',alg,'sparse','final','fix','prioritized','noper','controldiscreteadaptive']) , "seed_num":seednum, "color": "deepskyblue", "plot_name":  alg +" HER+PHiER(fix)+CL"})
        # exps.append({"exp_name": "_".join(['1017_A', 'Panda'+taskname+'-v3',alg,'sparse','noher','ama','fix','noper','nocl']) , "seed_num":seednum, "color": "palegreen", "plot_name":  alg +" HiER(ama)+CL"})
        # exps.append({"exp_name": "_".join(['1017_A', 'Panda'+taskname+'-v3',alg,'sparse','noher','ama','prioritized','noper','nocl']) , "seed_num":seednum, "color": "lightgreen", "plot_name":  alg +" PHiER(ama)+CL"})
        # exps.append({"exp_name": "_".join(['1017_A', 'Panda'+taskname+'-v3',alg,'sparse','noher','fix','fix','noper','nocl']) , "seed_num":seednum, "color": "blue", "plot_name":  alg +" HiER(fix)+CL"})
        # exps.append({"exp_name": "_".join(['1017_A', 'Panda'+taskname+'-v3',alg,'sparse','noher','fix','prioritized','noper','nocl']) , "seed_num":seednum, "color": "olive", "plot_name":  alg +" PHiER(fix)+CL"})

        # exps.append({"exp_name": "_".join(['1018_A', 'Panda'+taskname+'-v3',alg,'sparse','final','ama','fix','noper','controldiscreteadaptive']) , "seed_num":seednum, "color": "orange", "plot_name":  alg +" HER+HiER(old ama)+CL"})
        # # exps.append({"exp_name": "_".join(['1018_A', 'Panda'+taskname+'-v3',alg,'sparse','final','ama','prioritized','noper','controldiscreteadaptive']) , "seed_num":seednum, "color": "gold", "plot_name":  alg +" HER+PHiER(old ama)+CL"})
        
        # exps.append({"exp_name": "_".join(['1017_A', 'Panda'+taskname+'-v3',alg,'sparse','final','fix','fix','noper','controldiscreteadaptive']) , "seed_num":seednum, "color": "green", "plot_name":  alg +" HER+HiER(fix)+CL"})
        # # exps.append({"exp_name": "_".join(['1018_A', 'Panda'+taskname+'-v3',alg,'sparse','final','fix','prioritized','noper','controldiscreteadaptive']) , "seed_num":seednum, "color": "lime", "plot_name":  alg +" HER+PHiER(fix)+CL"})
        # exps.append({"exp_name": "_".join(['1018_A', 'Panda'+taskname+'-v3',alg,'sparse','noher','nohl','fix','proportional','controldiscreteadaptive']) , "seed_num":seednum, "color": "green", "plot_name":  alg +" PER+CL"})

        # # exps.append({"exp_name": "_".join(['1019_A', 'Panda'+taskname+'-v3',alg,'sparse','final','ama','fix','noper','controldiscreteadaptive']) , "seed_num":seednum, "color": "blue", "plot_name":  alg +" HER+HiER(ama)+CL"})
        # # exps.append({"exp_name": "_".join(['1019_A', 'Panda'+taskname+'-v3',alg,'sparse','final','ama','prioritized','noper','controldiscreteadaptive']) , "seed_num":seednum, "color": "aqua", "plot_name":  alg +" HER+PHiER(ama)+CL"})

        # # exps.append({"exp_name": "_".join(['1019_A', 'Panda'+taskname+'-v3',alg,'sparse','final','multifix','prioritized','noper','controldiscreteadaptive']) , "seed_num":seednum, "color": "magenta", "plot_name":  alg +" HER+PHiER(multifix)+CL"})

        # exps.append({"exp_name": "_".join(['1020_A', 'Panda'+taskname+'-v3',alg,'sparse','final','predefined','fix','noper','controldiscreteadaptive']) , "seed_num":seednum, "color": "blue", "plot_name":  alg +" HER+HiER(predefined)+CL"})
        # # exps.append({"exp_name": "_".join(['1020_A', 'Panda'+taskname+'-v3',alg,'sparse','final','predefined','prioritized','noper','controldiscreteadaptive']) , "seed_num":seednum, "color": "aqua", "plot_name":  alg +" HER+PHiER(predefined)+CL"})

        # #exps.append({"exp_name": "_".join(['1019_A', 'Panda'+taskname+'-v3',alg,'sparse','final','multifix','fix','noper','controldiscreteadaptive']) , "seed_num":seednum, "color": "purple", "plot_name":  alg +" HER+HiER(multifix)+CL"})

        # exps.append({"exp_name": "_".join(['1022_A', 'Panda'+taskname+'-v3',alg,'sparse','final','ama','prioritized','noper','controldiscreteadaptive']) , "seed_num":seednum, "color": "gold", "plot_name":  alg +" HER+PHiER(ama)+CL"})
        # exps.append({"exp_name": "_".join(['1022_A', 'Panda'+taskname+'-v3',alg,'sparse','final','fix','prioritized','noper','controldiscreteadaptive']) , "seed_num":seednum, "color": "lime", "plot_name":  alg +" HER+PHiER(fix)+CL"})
        # exps.append({"exp_name": "_".join(['1022_A', 'Panda'+taskname+'-v3',alg,'sparse','final','predefined','prioritized','noper','controldiscreteadaptive']) , "seed_num":seednum, "color": "aqua", "plot_name":  alg +" HER+PHiER(predefined)+CL"})
        # #exps.append({"exp_name": "_".join(['1022_A', 'Panda'+taskname+'-v3',alg,'sparse','final','multifix','prioritized','noper','controldiscreteadaptive']) , "seed_num":seednum, "color": "magenta", "plot_name":  alg +" HER+PHiER(multifix)+CL"})

        #exps.append({"exp_name": "_".join(['1025_A', 'Panda'+taskname+'-v3',alg,'sparse','final','predefined','fix','hbr01','noper','controldiscreteadaptive']) , "seed_num":seednum, "color": "darkcyan", "plot_name":  alg +" HER+HiER(fix 0.1)+CL"})
        #exps.append({"exp_name": "_".join(['1025_A', 'Panda'+taskname+'-v3',alg,'sparse','final','predefined','fix','hbr03','noper','controldiscreteadaptive']) , "seed_num":seednum, "color": "royalblue", "plot_name":  alg +" HER+HiER(fix 0.3)+CL"})
        # exps.append({"exp_name": "_".join(['1025_A', 'Panda'+taskname+'-v3',alg,'sparse','final','predefined','fix','hbr05','noper','controldiscreteadaptive']) , "seed_num":seednum, "color": "magenta", "plot_name":  alg +" HER+HiER(fix 0.5)+CL"})
        #exps.append({"exp_name": "_".join(['1025_A', 'Panda'+taskname+'-v3',alg,'sparse','final','predefined','fix','hbr07','noper','controldiscreteadaptive']) , "seed_num":seednum, "color": "navy", "plot_name":  alg +" HER+HiER(fix 0.7)+CL"})
        #exps.append({"exp_name": "_".join(['1025_A', 'Panda'+taskname+'-v3',alg,'sparse','final','predefined','fix','hbr09','noper','controldiscreteadaptive']) , "seed_num":seednum, "color": "indigo", "plot_name":  alg +" HER+HiER(fix 0.9)+CL"})

        #########################################################################################

        # # PUSH
        # exps.append({"exp_name": "_".join(['1025_A', 'Panda'+taskname+'-v3',alg,'sparse','final','predefined','fix','hbr05','noper','controldiscreteadaptive']) , "seed_num":seednum, "color": "magenta", "plot_name":  alg +" HER + CL + HiER"})
        # exps.append({"exp_name": "_".join(['1012_B', 'Panda'+taskname+'-v3',alg,'sparse','final','nohl','noper','nocl']) , "seed_num":seednum, "color": "orange", "plot_name": alg + " HER"}) 
        # exps.append({"exp_name": "_".join(['1018_A', 'Panda'+taskname+'-v3',alg,'sparse','noher','nohl','fix','proportional','controldiscreteadaptive']) , "seed_num":seednum, "color": "blue", "plot_name":  alg +" CL + PER"})
        # exps.append({"exp_name": "_".join(['1015_A', 'Panda'+taskname+'-v3',alg,'sparse','noher','nohl','noper','controldiscreteadaptive']) , "seed_num":seednum, "color": "green", "plot_name":  alg +" CL"})
        # exps.append({"exp_name": "_".join(['1012_B', 'Panda'+taskname+'-v3',alg,'sparse','noher','nohl','noper','nocl']) , "seed_num":seednum, "color": "red", "plot_name": alg}) 

        # SLIDE
        # exps.append({"exp_name": "_".join(['1025_A', 'Panda'+taskname+'-v3',alg,'sparse','final','predefined','fix','hbr05','noper','controldiscreteadaptive']) , "seed_num":seednum, "color": "magenta", "plot_name":  alg +" HER + CL + HiER"})
        # exps.append({"exp_name": "_".join(['1018_A', 'Panda'+taskname+'-v3',alg,'sparse','noher','nohl','fix','proportional','controldiscreteadaptive']) , "seed_num":seednum, "color": "blue", "plot_name":  alg +" CL + PER"})
        # exps.append({"exp_name": "_".join(['1015_A', 'Panda'+taskname+'-v3',alg,'sparse','noher','nohl','noper','controldiscreteadaptive']) , "seed_num":seednum, "color": "green", "plot_name":  alg +" CL"})
        # exps.append({"exp_name": "_".join(['1012_B', 'Panda'+taskname+'-v3',alg,'sparse','final','nohl','noper','nocl']) , "seed_num":seednum, "color": "orange", "plot_name": alg + " HER"})     
        # exps.append({"exp_name": "_".join(['1012_B', 'Panda'+taskname+'-v3',alg,'sparse','noher','nohl','noper','nocl']) , "seed_num":seednum, "color": "red", "plot_name": alg}) 

        #########################################################################################

        # exps.append({"exp_name": "_".join(['1012_B', 'Panda'+taskname+'-v3',alg,'sparse','final','nohl','noper','nocl']) , "seed_num":seednum, "color": "blue", "plot_name": alg + " HER 1"}) 
        # exps.append({"exp_name": "_".join(['1026_A', 'Panda'+taskname+'-v3',alg,'sparse','final','nohl','fix','noper','nocl']) , "seed_num":seednum, "color": "magenta", "plot_name": alg + " HER 2"}) 
        # exps.append({"exp_name": "_".join(['1026_A', 'Panda'+taskname+'-v3',alg,'sparse','noher','nohl','fix','noper','nocl']) , "seed_num":seednum, "color": "orange", "plot_name": alg})     

        # exps.append({"exp_name": "_".join(['1023_B', taskname+'-v4',alg,'sparse','noher','nohl','fix','noper','nocl']) , "seed_num":seednum, "color": "maroon", "plot_name":  alg +" baseline"})
        # exps.append({"exp_name": "_".join(['1024_B', taskname+'-v4',alg,'sparse','noher','fix','fix','noper','nocl']) , "seed_num":seednum, "color": "green", "plot_name":  alg +" HiER(fix)"})
        # exps.append({"exp_name": "_".join(['1024_B', taskname+'-v4',alg,'sparse','noher','fix','prioritized','noper','nocl']) , "seed_num":seednum, "color": "lime", "plot_name":  alg +" PHiER(fix)"})
        # exps.append({"exp_name": "_".join(['1024_B', taskname+'-v4',alg,'sparse','noher','ama','fix','noper','nocl']) , "seed_num":seednum, "color": "blue", "plot_name":  alg +" HiER(ama)"})
        # exps.append({"exp_name": "_".join(['1024_B', taskname+'-v4',alg,'sparse','noher','ama','prioritized','noper','nocl']) , "seed_num":seednum, "color": "aqua", "plot_name":  alg +" PHiER(ama)"})
        # exps.append({"exp_name": "_".join(['1024_B', taskname+'-v4',alg,'sparse','noher','amar','fix','noper','nocl']) , "seed_num":seednum, "color": "purple", "plot_name":  alg +" HiER(amar)"})
        # exps.append({"exp_name": "_".join(['1024_B', taskname+'-v4',alg,'sparse','noher','amar','prioritized','noper','nocl']) , "seed_num":seednum, "color": "magenta", "plot_name":  alg +" PHiER(amar)"})
        # exps.append({"exp_name": "_".join(['1024_B', taskname+'-v4',alg,'sparse','noher','predefined','fix','noper','nocl']) , "seed_num":seednum, "color": "orange", "plot_name":  alg +" HiER(predefined)"})
        # exps.append({"exp_name": "_".join(['1024_B', taskname+'-v4',alg,'sparse','noher','predefined','prioritized','noper','nocl']) , "seed_num":seednum, "color": "gold", "plot_name":  alg +" PHiER(predefined)"})

        # Buffer ##########################x
        # exps.append({"exp_name": "_".join(['1028_A', 'Panda'+taskname+'-v3',alg,'sparse','1e5','final','predefined','fix','noper','controldiscreteadaptive']) , "seed_num":seednum, "color": "cornflowerblue", "plot_name":  alg +" HER + CL + HiER + buffer 1e5"})
        # exps.append({"exp_name": "_".join(['1028_A', 'Panda'+taskname+'-v3',alg,'sparse','5e5','final','predefined','fix','noper','controldiscreteadaptive']) , "seed_num":seednum, "color": "blue", "plot_name":  alg +" HER + CL + HiER + buffer 5e5"})
        # exps.append({"exp_name": "_".join(['1028_A', 'Panda'+taskname+'-v3',alg,'sparse','final','predefined','fix','noper','controldiscreteadaptive']) , "seed_num":seednum, "color": "magenta", "plot_name":  alg +" HER + CL + HiER + buffer 1e6"})
        # exps.append({"exp_name": "_".join(['1028_A', 'Panda'+taskname+'-v3',alg,'sparse','2e6','final','predefined','fix','noper','controldiscreteadaptive']) , "seed_num":seednum, "color": "purple", "plot_name":  alg +" HER + CL + HiER + buffer 2e6"})
        
        # # Alpha ############################
        # exps.append({"exp_name": "_".join(['1028_A5', 'Panda'+taskname+'-v3',alg,'sparse','1e6','final','predefined','fix','noper','controldiscreteadaptive']) , "seed_num":seednum, "color": "cornflowerblue", "plot_name":  alg +" HER + CL + HiER + alpha 0.1"})
        # exps.append({"exp_name": "_".join(['1028_A', 'Panda'+taskname+'-v3',alg,'sparse','final','predefined','fix','noper','controldiscreteadaptive']) , "seed_num":seednum, "color": "magenta", "plot_name":  alg +" HER + CL + HiER + alpha 0.2"})
        # exps.append({"exp_name": "_".join(['1028_A6', 'Panda'+taskname+'-v3',alg,'sparse','1e6','final','predefined','fix','noper','controldiscreteadaptive']) , "seed_num":seednum, "color": "blue", "plot_name":  alg +" HER + CL + HiER + alpha 0.4"})
       
        # Gamma
        # exps.append({"exp_name": "_".join(['1028_A', 'Panda'+taskname+'-v3',alg,'sparse','final','predefined','fix','noper','controldiscreteadaptive']) , "seed_num":seednum, "color": "magenta", "plot_name":  alg +" HER + CL + HiER + gamma 0.95"})
        # exps.append({"exp_name": "_".join(['1028_A7', 'Panda'+taskname+'-v3',alg,'sparse','1e6','final','predefined','fix','noper','controldiscreteadaptive']) , "seed_num":seednum, "color": "blue", "plot_name":  alg +" HER + CL + HiER + gamma 0.99"})
      
        # Network
        # exps.append({"exp_name": "_".join(['1028_A', 'Panda'+taskname+'-v3',alg,'sparse','final','predefined','fix','noper','controldiscreteadaptive']) , "seed_num":seednum, "color": "magenta", "plot_name":  alg +" HER + CL + HiER + network(256,256,256)"})
        # exps.append({"exp_name": "_".join(['1028_A8', 'Panda'+taskname+'-v3',alg,'sparse','1e6','final','predefined','fix','noper','controldiscreteadaptive']) , "seed_num":seednum, "color": "purple", "plot_name":  alg +" HER + CL + HiER + network(512,512,512)"})
        # exps.append({"exp_name": "_".join(['1028_A9', 'Panda'+taskname+'-v3',alg,'sparse','1e6','final','predefined','fix','noper','controldiscreteadaptive']) , "seed_num":seednum, "color": "cornflowerblue", "plot_name":  alg +" HER + CL + HiER + network(128,128,128)"})
        # exps.append({"exp_name": "_".join(['1028_A10', 'Panda'+taskname+'-v3',alg,'sparse','1e6','final','predefined','fix','noper','controldiscreteadaptive']) , "seed_num":seednum, "color": "blue", "plot_name":  alg +" HER + CL + HiER + network(256,256,128)"})
        # exps.append({"exp_name": "_".join(['1028_A11', 'Panda'+taskname+'-v3',alg,'sparse','1e6','final','predefined','fix','noper','controldiscreteadaptive']) , "seed_num":seednum, "color": "navy", "plot_name":  alg +" HER + CL + HiER + network(256,256,128,64)"})
      
        # Alpha ############################
        # exps.append({"exp_name": "_".join(['1030_A', 'Panda'+taskname+'-v3',alg,'alp02','gam095','sparse','final','predefined','fix','noper','controldiscreteadaptive']) , "seed_num":seednum, "color": "red", "plot_name":  alg +" HER + CL + HiER + alpha 0.2 + gamma 0.95"})
        # exps.append({"exp_name": "_".join(['1030_A', 'Panda'+taskname+'-v3',alg,'alp01','gam095','sparse','final','predefined','fix','noper','controldiscreteadaptive']) , "seed_num":seednum, "color": "blue", "plot_name":  alg +" HER + CL + HiER + alpha 0.1 + gamma 0.95"})
        # exps.append({"exp_name": "_".join(['1030_A', 'Panda'+taskname+'-v3',alg,'alp005','gam095','sparse','final','predefined','fix','noper','controldiscreteadaptive']) , "seed_num":seednum, "color": "royalblue", "plot_name":  alg +" HER + CL + HiER + alpha 0.05 + gamma 0.95"})
        # exps.append({"exp_name": "_".join(['1030_A', 'Panda'+taskname+'-v3',alg,'alp001','gam095','sparse','final','predefined','fix','noper','controldiscreteadaptive']) , "seed_num":seednum, "color": "turquoise", "plot_name":  alg +" HER + CL + HiER + alpha 0.01 + gamma 0.95"})
      
        # exps.append({"exp_name": "_".join(['1030_A', 'Panda'+taskname+'-v3',alg,'alp02','gam09','sparse','final','predefined','fix','noper','controldiscreteadaptive']) , "seed_num":seednum, "color": "indigo", "plot_name":  alg +" HER + CL + HiER + alpha 0.2 + gamma 0.9"})
        # exps.append({"exp_name": "_".join(['1030_A', 'Panda'+taskname+'-v3',alg,'alp01','gam09','sparse','final','predefined','fix','noper','controldiscreteadaptive']) , "seed_num":seednum, "color": "purple", "plot_name":  alg +" HER + CL + HiER + alpha 0.1 + gamma 0.9"})
        # exps.append({"exp_name": "_".join(['1030_A', 'Panda'+taskname+'-v3',alg,'alp005','gam09','sparse','final','predefined','fix','noper','controldiscreteadaptive']) , "seed_num":seednum, "color": "deeppink", "plot_name":  alg +" HER + CL + HiER + alpha 0.05 + gamma 0.9"})
        # exps.append({"exp_name": "_".join(['1030_A', 'Panda'+taskname+'-v3',alg,'alp001','gam09','sparse','final','predefined','fix','noper','controldiscreteadaptive']) , "seed_num":seednum, "color": "pink", "plot_name":  alg +" HER + CL + HiER + alpha 0.01 + gamma 0.9"})
      
        # Gamma ############################

        # exps.append({"exp_name": "_".join(['1030_A', 'Panda'+taskname+'-v3',alg,'alp02','gam095','sparse','final','predefined','fix','noper','controldiscreteadaptive']) , "seed_num":seednum, "color": "red", "plot_name":  alg +" HER + CL + HiER + alpha 0.2 + gamma 0.95"})
        # exps.append({"exp_name": "_".join(['1030_A', 'Panda'+taskname+'-v3',alg,'alp02','gam09','sparse','final','predefined','fix','noper','controldiscreteadaptive']) , "seed_num":seednum, "color": "blue", "plot_name":  alg +" HER + CL + HiER + alpha 0.2 + gamma 0.9"})
        # exps.append({"exp_name": "_".join(['1030_A', 'Panda'+taskname+'-v3',alg,'alp02','gam08','sparse','final','predefined','fix','noper','controldiscreteadaptive']) , "seed_num":seednum, "color": "royalblue", "plot_name":  alg +" HER + CL + HiER + alpha 0.2 + gamma 0.8"})
        # exps.append({"exp_name": "_".join(['1030_A', 'Panda'+taskname+'-v3',alg,'alp02','gam07','sparse','final','predefined','fix','noper','controldiscreteadaptive']) , "seed_num":seednum, "color": "lightseagreen", "plot_name":  alg +" HER + CL + HiER + alpha 0.2 + gamma 0.7"})
        # exps.append({"exp_name": "_".join(['1030_A', 'Panda'+taskname+'-v3',alg,'alp02','gam06','sparse','final','predefined','fix','noper','controldiscreteadaptive']) , "seed_num":seednum, "color": "turquoise", "plot_name":  alg +" HER + CL + HiER + alpha 0.2 + gamma 0.6"})
      

        # exps.append({"exp_name": "_".join(['1030_A', 'Panda'+taskname+'-v3',alg,'alp01','gam095','sparse','final','predefined','fix','noper','controldiscreteadaptive']) , "seed_num":seednum, "color": "indigo", "plot_name":  alg +" HER + CL + HiER + alpha 0.1 + gamma 0.95"})
        # exps.append({"exp_name": "_".join(['1030_A', 'Panda'+taskname+'-v3',alg,'alp01','gam09','sparse','final','predefined','fix','noper','controldiscreteadaptive']) , "seed_num":seednum, "color": "purple", "plot_name":  alg +" HER + CL + HiER + alpha 0.1 + gamma 0.9"})
        # exps.append({"exp_name": "_".join(['1030_A', 'Panda'+taskname+'-v3',alg,'alp01','gam08','sparse','final','predefined','fix','noper','controldiscreteadaptive']) , "seed_num":seednum, "color": "deeppink", "plot_name":  alg +" HER + CL + HiER + alpha 0.1 + gamma 0.8"})
        # exps.append({"exp_name": "_".join(['1030_A', 'Panda'+taskname+'-v3',alg,'alp01','gam07','sparse','final','predefined','fix','noper','controldiscreteadaptive']) , "seed_num":seednum, "color": "hotpink", "plot_name":  alg +" HER + CL + HiER + alpha 0.1 + gamma 0.7"})
        # exps.append({"exp_name": "_".join(['1030_A', 'Panda'+taskname+'-v3',alg,'alp01','gam06','sparse','final','predefined','fix','noper','controldiscreteadaptive']) , "seed_num":seednum, "color": "pink", "plot_name":  alg +" HER + CL + HiER + alpha 0.1 + gamma 0.6"})
      
        # PickAndPlace ####################xx
        # exps.append({"exp_name": "_".join(['1015_A', 'Panda'+taskname+'-v3',alg,'sparse','noher','nohl','noper','controldiscreteadaptive']) , "seed_num":seednum, "color": "orange", "plot_name":  alg})
        # exps.append({"exp_name": "_".join(['1030_A', 'Panda'+taskname+'-v3',alg,'alp02','gam095','sparse','final','predefined','fix','noper','controldiscreteadaptive']) , "seed_num":seednum, "color": "blue", "plot_name":  alg +" HER + CL + HiER"})
        # exps.append({"exp_name": "_".join(['1030_A', 'Panda'+taskname+'-v3',alg,'alp02','gam095','state_change_bonus','final','predefined','fix','noper','controldiscreteadaptive']) , "seed_num":seednum, "color": "magenta", "plot_name":  alg +" HER + CL + HiER + REWBONUS"})
      
        # DDPG ###############################x
        # exps.append({"exp_name": "_".join(['1015_A', 'Panda'+taskname+'-v3',alg,'sparse','noher','nohl','noper','controldiscreteadaptive']) , "seed_num":seednum, "color": "orange", "plot_name":  alg})
        # exps.append({"exp_name": "_".join(['1030_A', 'Panda'+taskname+'-v3',alg,'alp02','gam095','sparse','final','predefined','fix','noper','controldiscreteadaptive']) , "seed_num":seednum, "color": "magenta", "plot_name":  alg +" HER + CL + HiER"})
      
        # Learning rate ######################x

        # exps.append({"exp_name": "_".join(['1101_A', 'Panda'+taskname+'-v3',alg,'lr001','sparse','final','predefined','fix','5e4','noper','controldiscreteadaptive']) , "seed_num":seednum, "color": "purple", "plot_name":  alg +" HER + CL + HiER + lr 0.01"})
        # exps.append({"exp_name": "_".join(['1101_A', 'Panda'+taskname+'-v3',alg,'lr0005','sparse','final','predefined','fix','5e4','noper','controldiscreteadaptive']) , "seed_num":seednum, "color": "violet", "plot_name":  alg +" HER + CL + HiER + lr 0.005"})
        # exps.append({"exp_name": "_".join(['1101_A', 'Panda'+taskname+'-v3',alg,'lr0002','sparse','final','predefined','fix','5e4','noper','controldiscreteadaptive']) , "seed_num":seednum, "color": "blue", "plot_name":  alg +" HER + CL + HiER + lr 0.002"})

        # exps.append({"exp_name": "_".join(['1030_A', 'Panda'+taskname+'-v3',alg,'alp01','gam095','sparse','final','predefined','fix','noper','controldiscreteadaptive']) , "seed_num":seednum, "color": "red", "plot_name":  alg +" HER + CL + HiER + lr 0.001"})

        # exps.append({"exp_name": "_".join(['1101_A', 'Panda'+taskname+'-v3',alg,'lr00005','sparse','final','predefined','fix','5e4','noper','controldiscreteadaptive']) , "seed_num":seednum, "color": "cornflowerblue", "plot_name":  alg +" HER + CL + HiER + lr 0.0005"})
        # exps.append({"exp_name": "_".join(['1101_A', 'Panda'+taskname+'-v3',alg,'lr00001','sparse','final','predefined','fix','5e4','noper','controldiscreteadaptive']) , "seed_num":seednum, "color": "aquamarine", "plot_name":  alg +" HER + CL + HiER + lr 0.0001"})
        # exps.append({"exp_name": "_".join(['1101_A', 'Panda'+taskname+'-v3',alg,'lr5e-05','sparse','final','predefined','fix','5e4','noper','controldiscreteadaptive']) , "seed_num":seednum, "color": "springgreen", "plot_name":  alg +" HER + CL + HiER + lr 0.00005"})        
      
        # Highlight buffer size ######################x

        # exps.append({"exp_name": "_".join(['1101_A', 'Panda'+taskname+'-v3',alg,'lr0001','sparse','final','predefined','fix','1e6','noper','controldiscreteadaptive']) , "seed_num":seednum, "color": "purple", "plot_name":  alg +" HER + CL + HiER(buffer_size: 1e6)"})
        # exps.append({"exp_name": "_".join(['1101_A', 'Panda'+taskname+'-v3',alg,'lr0001','sparse','final','predefined','fix','5e5','noper','controldiscreteadaptive']) , "seed_num":seednum, "color": "violet", "plot_name":  alg +" HER + CL + HiER(buffer_size: 5e5)"})
        # exps.append({"exp_name": "_".join(['1101_A', 'Panda'+taskname+'-v3',alg,'lr0001','sparse','final','predefined','fix','1e5','noper','controldiscreteadaptive']) , "seed_num":seednum, "color": "blue", "plot_name":  alg +" HER + CL + HiER(buffer_size: 1e5)"})

        # exps.append({"exp_name": "_".join(['1030_A', 'Panda'+taskname+'-v3',alg,'alp01','gam095','sparse','final','predefined','fix','noper','controldiscreteadaptive']) , "seed_num":seednum, "color": "red", "plot_name":  alg +" HER + CL + HiER(buffer_size: 5e4)"})

        # exps.append({"exp_name": "_".join(['1101_A', 'Panda'+taskname+'-v3',alg,'lr0001','sparse','final','predefined','fix','1e4','noper','controldiscreteadaptive']) , "seed_num":seednum, "color": "cornflowerblue", "plot_name":  alg +" HER + CL + HiER(buffer_size: 1e4)"})
        # exps.append({"exp_name": "_".join(['1101_A', 'Panda'+taskname+'-v3',alg,'lr0001','sparse','final','predefined','fix','5e3','noper','controldiscreteadaptive']) , "seed_num":seednum, "color": "aquamarine", "plot_name":  alg +" HER + CL + HiER(buffer_size: 5e3)"})
        # exps.append({"exp_name": "_".join(['1101_A', 'Panda'+taskname+'-v3',alg,'lr0001','sparse','final','predefined','fix','1e3','noper','controldiscreteadaptive']) , "seed_num":seednum, "color": "springgreen", "plot_name":  alg +" HER + CL + HiER(buffer_size: 1e3)"})

        # SOTA ######################x

        # exps.append({"exp_name": "_".join(['1102_A', 'Panda'+taskname+'-v3',alg,'sparse','noher','nohl','fix','noper','nocl']) , "seed_num":seednum, "color": "brown", "plot_name":  alg})
        # exps.append({"exp_name": "_".join(['1102_A', 'Panda'+taskname+'-v3',alg,'sparse','final','nohl','fix','noper','nocl']) , "seed_num":seednum, "color": "orange", "plot_name":  alg + " + HER"})
        # exps.append({"exp_name": "_".join(['1102_A', 'Panda'+taskname+'-v3',alg,'sparse','noher','nohl','fix','proportional','nocl']) , "seed_num":seednum, "color": "green", "plot_name":  alg + " + PER"})
        # exps.append({"exp_name": "_".join(['1102_A', 'Panda'+taskname+'-v3',alg,'sparse','noher','nohl','fix','noper','controldiscreteadaptive']) , "seed_num":seednum, "color": "blue", "plot_name":  alg + " + CL"})
        # exps.append({"exp_name": "_".join(['1102_A', 'Panda'+taskname+'-v3',alg,'sparse','final','nohl','fix','noper','controldiscreteadaptive']) , "seed_num":seednum, "color": "gold", "plot_name":  alg + " + HER + CL"})
        # exps.append({"exp_name": "_".join(['1102_A', 'Panda'+taskname+'-v3',alg,'sparse','noher','nohl','fix','proportional','controldiscreteadaptive']) , "seed_num":seednum, "color": "lawngreen", "plot_name":  alg + " + PER + CL"})
        # exps.append({"exp_name": "_".join(['1102_A', 'Panda'+taskname+'-v3',alg,'sparse','noher','predefined','fix','noper','nocl']) , "seed_num":seednum, "color": "springgreen", "plot_name":  alg + " + HiER"})
        # exps.append({"exp_name": "_".join(['1102_A', 'Panda'+taskname+'-v3',alg,'sparse','final','predefined','fix','noper','nocl']) , "seed_num":seednum, "color": "aqua", "plot_name":  alg + " + HER + HiER"})
        # exps.append({"exp_name": "_".join(['1102_A', 'Panda'+taskname+'-v3',alg,'sparse','noher','predefined','fix','noper','controldiscreteadaptive']) , "seed_num":seednum, "color": "navy", "plot_name":  alg + " + CL + HiER"})
        # exps.append({"exp_name": "_".join(['1102_A', 'Panda'+taskname+'-v3',alg,'sparse','final','predefined','fix','noper','controldiscreteadaptive']) , "seed_num":seednum, "color": "magenta", "plot_name":  alg + " + HER + CL + HiER"})
        # #exps.append({"exp_name": "_".join(['1102_A', 'Panda'+taskname+'-v3',alg,'sparse','noher','predefined','fix','proportional','controldiscreteadaptive']) , "seed_num":seednum, "color": "hotpink", "plot_name":  alg + " + PER + CL + HiER"})
        # #exps.append({"exp_name": "_".join(['1102_A', 'Panda'+taskname+'-v3',alg,'sparse','final','predefined','fix','proportional','controldiscreteadaptive']) , "seed_num":seednum, "color": "indigo", "plot_name":  alg + " + HER + PER + CL + HiER"})


        # CL ###########################
              
        # exps.append({"exp_name": "_".join(['1026_A', 'Panda'+taskname+'-v3',alg,'sparse','final','nohl','fix','noper','predefined_linear']) , "seed_num":seednum, "color": "blue", "plot_name":  alg +" HER + CL(predefined_linear)"})
        exps.append({"exp_name": "_".join(['1026_A', 'Panda'+taskname+'-v3',alg,'sparse','final','predefined','fix','noper','predefined_linear']) , "seed_num":seednum, "color": "orange", "plot_name":  "predefined"})

        # exps.append({"exp_name": "_".join(['1026_A', 'Panda'+taskname+'-v3',alg,'sparse','final','nohl','fix','noper','predefinedtwostage_linear']) , "seed_num":seednum, "color": "green", "plot_name":  alg +" HER + CL(predefinedtwostage_linear)"})
        exps.append({"exp_name": "_".join(['1026_A', 'Panda'+taskname+'-v3',alg,'sparse','final','predefined','fix','noper','predefinedtwostage_linear']) , "seed_num":seednum, "color": "gold", "plot_name":  "predefined 2-stage"})

        # exps.append({"exp_name": "_".join(['1026_A', 'Panda'+taskname+'-v3',alg,'sparse','final','nohl','fix','noper','selfpaced']) , "seed_num":seednum, "color": "gold", "plot_name":  alg +" HER + CL(selfpaced)"})
        exps.append({"exp_name": "_".join(['1026_A', 'Panda'+taskname+'-v3',alg,'sparse','final','predefined','fix','noper','selfpaced']) , "seed_num":seednum, "color": "springgreen", "plot_name":  "self-paced"})

        # exps.append({"exp_name": "_".join(['1026_A', 'Panda'+taskname+'-v3',alg,'sparse','final','nohl','fix','noper','selfpaceddual']) , "seed_num":seednum, "color": "orange", "plot_name":  alg +" HER + CL(selfpaceddual)"})
        exps.append({"exp_name": "_".join(['1026_A', 'Panda'+taskname+'-v3',alg,'sparse','final','predefined','fix','noper','selfpaceddual']) , "seed_num":seednum, "color": "forestgreen", "plot_name":  "self-paced dual"})

        # exps.append({"exp_name": "_".join(['1026_A', 'Panda'+taskname+'-v3',alg,'sparse','final','nohl','fix','noper','controldiscrete_const']) , "seed_num":seednum, "color": "red", "plot_name":  alg +" HER + CL(controldiscrete_const)"})
        exps.append({"exp_name": "_".join(['1026_A', 'Panda'+taskname+'-v3',alg,'sparse','final','predefined','fix','noper','controldiscrete_const']) , "seed_num":seednum, "color": "royalblue", "plot_name":  "control"})

        # exps.append({"exp_name": "_".join(['1028_A', 'Panda'+taskname+'-v3',alg,'sparse','final','nohl','fix','noper','controldiscreteadaptive']) , "seed_num":seednum, "color": "aqua", "plot_name":  alg +" HER + CL(controldiscreteadaptive)"})
        exps.append({"exp_name": "_".join(['1028_A', 'Panda'+taskname+'-v3',alg,'sparse','final','predefined','fix','noper','controldiscreteadaptive']) , "seed_num":seednum, "color": "blue", "plot_name":  "control adaptive"})




        exp_test_color_list = []
        exp_name_list = []
        for i in range(len(exps)):
            exp_test_color_list.append(exps[i]['color'])
            exp_name_list.append(exps[i]['plot_name'])
            

        for plotdata,plotdata_y in zip(plotdata_list,plotdata_y_list):

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
                    pivot['t'] = pivot['t'] / 1000
                    pivot = pivot[['exp_name','seed','t','value','plot_name']]

                    data_pd = data_pd.append(pivot, ignore_index = True)

                        
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
                                        'plot_name': pivot.iloc[0]["plot_name"]}
                        maxdata_pd = maxdata_pd.append(df_append, ignore_index = True)
            
                    maxdata_pd['HER'] = False
                    maxdata_pd['PER'] = False
                    maxdata_pd['CL'] = False
                    maxdata_pd['HiER'] = False

                    for ind in maxdata_pd.index:
                        if maxdata_pd['exp_name'][ind].find('noher') == -1: maxdata_pd['HER'][ind] = True
                        if maxdata_pd['exp_name'][ind].find('noper') == -1: maxdata_pd['PER'][ind] = True
                        if maxdata_pd['exp_name'][ind].find('nocl') == -1: maxdata_pd['CL'][ind] = True
                        if maxdata_pd['exp_name'][ind].find('nohl') == -1: maxdata_pd['HiER'][ind] = True
                
                    maxdata_pd = maxdata_pd[['HER','PER','CL','HiER','meanmaxvalue','stdmaxvalue']]

                    print(maxdata_pd.to_string())
                    print(maxdata_pd.head())

                    maxdata_pd.to_csv(os.path.join(current_dir, args.outdir, logdir, plotdata+".csv"),index=False)

            # Separate plotting ########################## 

            fig, ax = plt.subplots(figsize=(10,8))
            #exps_names = data_pd['exp_name'].unique()

            for i in range(len(exps)):
                for j in range(exps[i]['seed_num']):
                    pivot=data_pd[data_pd["exp_name"] == exps[i]['exp_name']]  
                    pivot=pivot[pivot["seed"] == j]
                    plt.plot(pivot['t'],pivot['value'],color=exps[i]['color'],label=exps[i]['plot_name']) if j == 0 else plt.plot(pivot['t'],pivot['value'],color=exps[i]['color'])
                    

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

            sns.lineplot(data=data_pd, x="t", y="value", hue="plot_name", errorbar=('ci', 95), palette=exp_test_color_list, legend=args.legend)

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

