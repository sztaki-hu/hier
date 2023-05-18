import os
import argparse


exps = ["0420_A","0420_B","0420_C"]
trainids = [3,3,3]

# exps = ["0420_A"]
# trainids = [1]

for i in range(len(exps)):
    for j in range(int(trainids[i])):
        configfile = "logs/" + exps[i] + "_stack_blocks_sac/"
        print("python3 test_epochs.py --configfile " + configfile + " --trainid " + str(j))
        print(" EXP NUM: " + str(i+1) + " / " +  str(len(exps)) + " | Train id: " + str(j+1) + " / " + str(trainids[i]))
        os.system("python3 test_epochs.py --configfile " + configfile + " --trainid " + str(j))

