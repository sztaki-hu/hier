
from typing import Dict, Union

from rltrain.buffers.replay import ReplayBuffer 
from rltrain.buffers.prioritized_replay import PrioritizedReplay

def make_per(config: Dict, config_framework: Dict) -> Union[ReplayBuffer, PrioritizedReplay]:

    per_mode = config['buffer']['per']['mode']
    print(per_mode)

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
        raise ValueError("[PER]: per_mode: '" + str(per_mode) + "' must be in : " + str(config_framework['per']['mode_list']))

    return replay_buffer