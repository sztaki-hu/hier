import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def create_folder(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)
        print(path + ' folder is created!')
    else:
        print(path + ' folder already exists!')

def generate_latex_table_3(data_array, tasks):
    # Start LaTeX table
    latex_code =  r"\begin{table*}[t]" + "\n"
    latex_code += r"    \centering" + "\n"
    latex_code += r"    \begin{tabular}{c|c|cccccc}" + "\n"
    latex_code += r"          & & \multicolumn{6}{c}{\texttt{"+tasks[0]+r"} | \texttt{"+tasks[1]+r"} | \texttt{"+tasks[2]+r"} } \\" + "\n"
    latex_code += r"          & \rotatebox{90}{HER} \rotatebox{90}{PER} \rotatebox{90}{ISE} \rotatebox{90}{HiER} & Mean $\uparrow$ & Median $\uparrow$ & IQM $\uparrow$ & OG $\downarrow$ & Max $\uparrow$ & Std $\downarrow$ \\" + "\n"
    latex_code += r"   \hline" + "\n"
    
    # Other sections
    sections = ["Baselines","HiER", "ISE", "HiER+"]
    j = 0
    for i in range(len(data_array)):
        if i in [4,8]:
            latex_code += r"   \hline" + "\n"
            latex_code += r"   \hline" + "\n"
        elif i == 12:
            latex_code += r"   \hline" + "\n"
        if i%4==0:
            latex_code += r"    \multirow{4}{*}{\rotatebox{90}{"+ sections[j] + "} } "
            j+=1

        latex_code += " & " + " & ".join(data_array[i]) + r" \\" + "\n"
    
    # End LaTeX table
    latex_code += r"    \end{tabular}" + "\n"
    latex_code += r"\end{table*}" + "\n"
    
    return latex_code

def generate_latex_table_2(data_array, tasks):
    # Start LaTeX table
    latex_code =  r"\begin{table*}[t]" + "\n"
    latex_code += r"    \centering" + "\n"
    latex_code += r"    \begin{tabular}{c|c|cccccc}" + "\n"
    latex_code += r"          & & \multicolumn{6}{c}{\texttt{"+tasks[0]+r"} | \texttt{"+tasks[1] +r"} } \\" + "\n"
    latex_code += r"          & \rotatebox{90}{HER} \rotatebox{90}{PER} \rotatebox{90}{ISE} \rotatebox{90}{HiER} & Mean $\uparrow$ & Median $\uparrow$ & IQM $\uparrow$ & OG $\downarrow$ & Max $\uparrow$ & Std $\downarrow$ \\" + "\n"
    latex_code += r"   \hline" + "\n"
    
    # Other sections
    sections = ["Baselines","HiER", "ISE", "HiER+"]
    j = 0
    for i in range(len(data_array)):
        if i in [4,8]:
            latex_code += r"   \hline" + "\n"
            latex_code += r"   \hline" + "\n"
        elif i == 12:
            latex_code += r"   \hline" + "\n"
        if i%4==0:
            latex_code += r"    \multirow{4}{*}{\rotatebox{90}{"+ sections[j] + "} } "
            j+=1

        latex_code += " & " + " & ".join(data_array[i]) + r" \\" + "\n"
    
    # End LaTeX table
    latex_code += r"    \end{tabular}" + "\n"
    latex_code += r"\end{table*}" + "\n"
    
    return latex_code

def generate_latex_table_1(data_array, tasks):
    # Start LaTeX table
    latex_code =  r"\begin{table*}[t]" + "\n"
    latex_code += r"    \centering" + "\n"
    latex_code += r"    \begin{tabular}{c|c|cccccc}" + "\n"
    latex_code += r"          & & \multicolumn{6}{c}{\texttt{"+tasks[0]+r"} } \\" + "\n"
    latex_code += r"          & \rotatebox{90}{HER} \rotatebox{90}{PER} \rotatebox{90}{ISE} \rotatebox{90}{HiER} & Mean $\uparrow$ & Median $\uparrow$ & IQM $\uparrow$ & OG $\downarrow$ & Max $\uparrow$ & Std $\downarrow$ \\" + "\n"
    latex_code += r"   \hline" + "\n"
    
    # Other sections
    sections = ["Baselines","HiER", "ISE", "HiER+"]
    j = 0
    for i in range(len(data_array)):
        if i in [4,8]:
            latex_code += r"   \hline" + "\n"
            latex_code += r"   \hline" + "\n"
        elif i == 12:
            latex_code += r"   \hline" + "\n"
        if i%4==0:
            latex_code += r"    \multirow{4}{*}{\rotatebox{90}{"+ sections[j] + "} } "
            j+=1

        latex_code += " & " + " & ".join(data_array[i]) + r" \\" + "\n"
    
    # End LaTeX table
    latex_code += r"    \end{tabular}" + "\n"
    latex_code += r"\end{table*}" + "\n"
    
    return latex_code

