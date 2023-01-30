import os

config_list_name = "/cfg/lists/A/"

for i in range(3):
    #print("python3 main.py --configfile " + config_list_name + str(i+1) + ".yaml")
    os.system("python3 main.py --configfile " + config_list_name + str(i+1) + ".yaml")
