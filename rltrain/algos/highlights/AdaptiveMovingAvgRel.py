import numpy as np
import collections

from rltrain.algos.highlights.HL import HL

class AdaptiveMovingAvgRelHL(HL):
    def __init__(self, config):
        super(AdaptiveMovingAvgRelHL, self).__init__(config)

        self.hl_threshold_start = config['buffer']['highlights']['amar']['threshold_start']
        self.hl_threshold_margin_relative = config['buffer']['highlights']['amar']['threshold_margin_relative']
        assert self.hl_threshold_margin_relative > 0
        self.hl_window = config['buffer']['highlights']['amar']['window']

        self.hl_threshold = self.hl_threshold_start
        self.hl_ep_rew_dq = collections.deque(maxlen=self.hl_window)
    
    def store_episode(self,episode,info_success,t):

        if self.hl_success_cond and info_success == False: return
        
        sum_rew = 0
        for (o, a, r, o2, d) in episode:
            sum_rew += r

        if sum_rew >= self.hl_threshold:
            for (o, a, r, o2, d) in episode:
                self.hl_replay_buffer.store(o, a, r, o2, d)

            self.hl_ep_rew_dq.append(sum_rew)
            if len(self.hl_ep_rew_dq) == self.hl_window: 
                margin = abs(np.mean(self.hl_ep_rew_dq) * self.hl_threshold_margin_relative)
                self.hl_threshold = np.mean(self.hl_ep_rew_dq) + margin
