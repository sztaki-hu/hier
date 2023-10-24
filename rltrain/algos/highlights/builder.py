
HIGHLIGHTS_MODES = ['nohl','fix','ama','amar','predefined','multifix']

def make_hl(config):

    hl_mode = config['buffer']['highlights']['mode']
    print(hl_mode)
    assert hl_mode in HIGHLIGHTS_MODES
    
    if hl_mode == 'nohl':
        from rltrain.algos.highlights.noHL import noHL as HL
    elif hl_mode == 'fix':
        from rltrain.algos.highlights.FixHL import FixHL as HL
    elif hl_mode == 'ama':
        from rltrain.algos.highlights.AdaptiveMovingAvg import AdaptiveMovingAvgHL as HL
    elif hl_mode == 'amar':
        from rltrain.algos.highlights.AdaptiveMovingAvgRel import AdaptiveMovingAvgRelHL as HL
    elif hl_mode == 'predefined':
        from rltrain.algos.highlights.PredefinedHL import PredefinedHL as HL
    elif hl_mode == 'multifix':
        from rltrain.algos.highlights.MultiFixHL import MultiFixHL as HL

    return HL(config)