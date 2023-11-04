from typing import Dict, Union

from rltrain.algos.hier.noHiER import noHiER
from rltrain.algos.hier.fixHiER import fixHiER
from rltrain.algos.hier.multifixHiER import multifixHiER
from rltrain.algos.hier.predefinedHiER import predefinedHiER
from rltrain.algos.hier.amaHiER import amaHiER
from rltrain.algos.hier.amarHiER import amarHiER

LAMBDA_MODES = ['nohier', 'fix', 'multifix', 'predefined', 'ama', 'amar']

def make_hier(config: Dict) -> Union[noHiER, fixHiER, multifixHiER, predefinedHiER, amaHiER, amarHiER]:

    lambda_mode = config['buffer']['hier']['lambda']['mode']
    print(lambda_mode)
    assert lambda_mode in LAMBDA_MODES
    
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
        assert False
    


    