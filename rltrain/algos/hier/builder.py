from typing import Dict, Union

from rltrain.algos.hier.noHiER import noHiER
from rltrain.algos.hier.fixHiER import fixHiER
from rltrain.algos.hier.multifixHiER import multifixHiER
from rltrain.algos.hier.predefinedHiER import predefinedHiER
from rltrain.algos.hier.amaHiER import amaHiER
from rltrain.algos.hier.amarHiER import amarHiER

def make_hier(config: Dict, config_framework: Dict) -> Union[noHiER, fixHiER, multifixHiER, predefinedHiER, amaHiER, amarHiER]:

    lambda_mode = config['buffer']['hier']['lambda']['mode']
    print(lambda_mode)

    xi_mode = config['buffer']['hier']['xi']['mode']
    print(xi_mode)
    
    if xi_mode not in config_framework['hier']['xi_mode_list']:
        raise ValueError("[HiER]: xi_mode: '" + str(xi_mode) + "' must be in : " + str(config_framework['hier']['xi_mode_list']))
    
    
    if lambda_mode == 'nohier':
        return noHiER(config)
    elif lambda_mode == 'fix':
        return fixHiER(config)
    elif lambda_mode == 'ama':
        return amaHiER(config)
    elif lambda_mode == 'amar':   
        return amarHiER(config)
    elif lambda_mode == 'predefined':       
        return predefinedHiER(config)
    elif lambda_mode == 'multifix':  
        return multifixHiER(config)
    else:
        raise ValueError("[HiER]: lambda_mode: '" + str(lambda_mode) + "' must be in : " + str(config_framework['hier']['lambda_mode_list']))
    


    