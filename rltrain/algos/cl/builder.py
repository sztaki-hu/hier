
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

CL_TYPES = ['nocl','nullcl','predefined','predefined2stage','predefined3stage',
            'selfpaced','control', 'controladaptive']

def make_cl(config: Dict, taskenv: GymPanda
            ) -> Union[noCL, nullCL, predefinedCL, predefined2stageCL, predefined3stageCL, 
                       selfpacedCL, controlCL, controladaptiveCL]:

    cl_mode = config['trainer']['cl']['type']
    print(cl_mode)
    assert cl_mode in CL_TYPES
    
    if cl_mode == 'nocl':
        return noCL(config, taskenv)
    elif cl_mode == 'nullcl':
        return nullCL(config, taskenv)
    elif cl_mode == 'predefined':
        return predefinedCL(config, taskenv)
    elif cl_mode == 'predefined2stage':
        return predefined2stageCL(config, taskenv)
    elif cl_mode == 'predefined3stage':
        return predefined3stageCL(config, taskenv)
    elif cl_mode == 'selfpaced':
        return selfpacedCL(config, taskenv)
    elif cl_mode == 'control':
        return controlCL(config, taskenv)
    elif cl_mode == 'controladaptive':
        return controladaptiveCL(config, taskenv)
    else:
        assert False

