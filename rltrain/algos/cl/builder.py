

CL_TYPES = ['nocl','nullcl','predefined','predefinedtwostage','predefinedthreestage',
            'selfpaced','selfpaceddual','controldiscrete', 'controldiscreteadaptive', 
            'examplebyexample']

def make_cl(config, env, replay_buffer):

    cl_mode = config['trainer']['cl']['type']
    print(cl_mode)
    assert cl_mode in CL_TYPES
    
    if cl_mode == 'nocl':
        from rltrain.algos.cl.NoCL import NoCL as CL
    elif cl_mode == 'nullcl':
        from rltrain.algos.cl.NullCL import NullCL as CL
    elif cl_mode == 'predefined':
        from rltrain.algos.cl.PredefinedCL import PredefinedCL as CL
    elif cl_mode == 'predefinedtwostage':
        from rltrain.algos.cl.PredefinedTwostageCL import PredefinedTwostageCL as CL
    elif cl_mode == 'predefinedthreestage':
        from rltrain.algos.cl.PredefinedThreestageCL import PredefinedThreestageCL as CL
    elif cl_mode == 'selfpaced':
        from rltrain.algos.cl.SelfPacedCL import SelfPacedCL as CL
    elif cl_mode == 'selfpaceddual':
        from rltrain.algos.cl.SelfPacedDualCL import SelfPacedDualCL as CL
    elif cl_mode == 'controldiscrete':
        from rltrain.algos.cl.ControlDiscreteCL import ControlDiscreteCL as CL
    elif cl_mode == 'controldiscreteadaptive':
        from rltrain.algos.cl.ControlDiscreteAdaptiveCL import ControlDiscreteAdaptiveCL as CL

    return CL(config, env, replay_buffer)