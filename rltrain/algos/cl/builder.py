
from typing import Dict, Union

from rltrain.taskenvs.GymPanda import GymPanda

from rltrain.algos.cl.noCL import noCL
from rltrain.algos.cl.nullCL import nullCL
from rltrain.algos.cl.predefinedCL import predefinedCL 
from rltrain.algos.cl.predefined2stageCL import predefined2stageCL 
from rltrain.algos.cl.predefined3stageCL import predefined3stageCL 
from rltrain.algos.cl.selfpacedCL import selfpacedCL 
from rltrain.algos.cl.controlCL import controlCL 
from rltrain.algos.cl.controladaptive import controladaptiveCL 

def make_cl(config: Dict, config_framework: Dict, taskenv: GymPanda
            ) -> Union[noCL, nullCL, predefinedCL, predefined2stageCL, predefined3stageCL, 
                       selfpacedCL, controlCL, controladaptiveCL]:

    c_mode = config['trainer']['cl']['type']
    print(c_mode)
    #assert c_mode in config_framework['cl']['cl_mode_list']
    
    if c_mode == 'nocl':
        return noCL(config, taskenv)
    elif c_mode == 'nullcl':
        return nullCL(config, taskenv)
    elif c_mode == 'predefined':
        return predefinedCL(config, taskenv)
    elif c_mode == 'predefined2stage':
        return predefined2stageCL(config, taskenv)
    elif c_mode == 'predefined3stage':
        return predefined3stageCL(config, taskenv)
    elif c_mode == 'selfpaced':
        return selfpacedCL(config, taskenv)
    elif c_mode == 'control':
        return controlCL(config, taskenv)
    elif c_mode == 'controladaptive':
        return controladaptiveCL(config, taskenv)
    else:
        raise ValueError("[CL]: c_mode: '" + str(c_mode) + "' must be in : " + str(config_framework['cl']['c_mode_list']))
   

