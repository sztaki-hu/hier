
PER_TYPES = ['noper','proportional']

def make_per(config):

    per_mode = config['buffer']['per']['mode']
    print(per_mode)
    assert per_mode in PER_TYPES

    if per_mode == 'noper':
        from rltrain.buffers.replay import ReplayBuffer 
        replay_buffer = ReplayBuffer(
            obs_dim=int(config['environment']['obs_dim']), 
            act_dim=int(config['environment']['act_dim']), 
            size=int(float(config['buffer']['replay_buffer_size'])))
    elif per_mode == 'proportional':
        from rltrain.buffers.prioritized_replay import PrioritizedReplay
        replay_buffer = PrioritizedReplay(
            size = int(float(config['buffer']['replay_buffer_size'])), 
            alpha = float(config['buffer']['per']['alpha']),
            beta_start = float(config['buffer']['per']['beta_start']),
            beta_frames = float(config['buffer']['per']['beta_frames_ratio']) * float(config['trainer']['total_timesteps']))

    return replay_buffer