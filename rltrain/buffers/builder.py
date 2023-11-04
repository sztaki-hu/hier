
from typing import Dict, Union

from rltrain.buffers.replay import ReplayBuffer 
from rltrain.buffers.prioritized_replay import PrioritizedReplay

PER_TYPES = ['noper','proportional']

def make_per(config: Dict) -> Union[ReplayBuffer, PrioritizedReplay]:

    per_mode = config['buffer']['per']['mode']
    print(per_mode)
    assert per_mode in PER_TYPES

    if per_mode == 'noper':
        replay_buffer = ReplayBuffer(
            obs_dim=int(config['environment']['obs_dim']), 
            act_dim=int(config['environment']['act_dim']), 
            size=int(float(config['buffer']['replay_buffer_size'])))
    elif per_mode == 'proportional':
        replay_buffer = PrioritizedReplay(
            size = int(float(config['buffer']['replay_buffer_size'])), 
            alpha = float(config['buffer']['per']['alpha']),
            beta_start = float(config['buffer']['per']['beta_start']),
            beta_frames = int(float(config['buffer']['per']['beta_frames_ratio']) * float(config['trainer']['total_timesteps'])))
    else:
        assert False

    return replay_buffer