
from typing import Dict, Union

from rltrain.taskenvs.GymPanda import GymPanda
from rltrain.taskenvs.GymMaze import GymMaze
from rltrain.taskenvs.GymFetch import GymFetch

from rltrain.algos.initstate.ise.MaxISE import MaxISE
from rltrain.algos.initstate.ise.MinISE import MinISE
from rltrain.algos.initstate.ise.PredefinedISE import PredefinedISE 
from rltrain.algos.initstate.ise.PredefinedTwoStageISE import PredefinedTwoStageISE 
from rltrain.algos.initstate.ise.PredefinedThreeStageISE import PredefinedThreeStageISE 
from rltrain.algos.initstate.ise.SelfPacedISE import SelfPacedISE 
from rltrain.algos.initstate.ise.ControlISE import ControlISE 
from rltrain.algos.initstate.ise.ControlAdaptiveISE import ControlAdaptiveISE 

from rltrain.algos.initstate.isedisc.PredefinedDiscISE import PredefinedDiscISE 

def make_ise(config: Dict, 
             config_framework: Dict, 
             taskenv: Union[GymPanda, GymMaze, GymFetch], 
            ) -> Union[MaxISE, MinISE, PredefinedISE, PredefinedTwoStageISE, PredefinedThreeStageISE, 
                       SelfPacedISE, ControlISE, ControlAdaptiveISE,
                       PredefinedDiscISE]:

    ise_mode = config['trainer']['init_state']['type']
    print(ise_mode)
    
    if ise_mode == 'max':
        return MaxISE(config, taskenv)
    elif ise_mode == 'min':
        return MinISE(config, taskenv)
    elif ise_mode == 'predefined':
        return PredefinedISE(config, taskenv)
    elif ise_mode == 'predefined2stage':
        return PredefinedTwoStageISE(config, taskenv)
    elif ise_mode == 'predefined3stage':
        return PredefinedThreeStageISE(config, taskenv)
    elif ise_mode == 'selfpaced':
        return SelfPacedISE(config, taskenv)
    elif ise_mode == 'control':
        return ControlISE(config, taskenv)
    elif ise_mode == 'controladaptive':
        return ControlAdaptiveISE(config, taskenv)
    elif ise_mode == 'predefined_disc':
        return PredefinedDiscISE(config, taskenv)
    else:
        raise ValueError("[ISE]: ise_mode: '" + str(ise_mode) + "' must be in : " + str(config_framework['ise']['mode_list']))
   

