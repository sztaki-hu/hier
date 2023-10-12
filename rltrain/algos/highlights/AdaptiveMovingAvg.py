import numpy as np
import collections

from rltrain.algos.highlights.HL import HL

class AdaptiveMovingAvgHL(HL):
    def __init__(self, config):
        super(AdaptiveMovingAvgHL, self).__init__(config)

        self.hl_threshold_start = config['buffer']['highlights']['ama']['threshold_start']
        self.hl_threshold_margin = config['buffer']['highlights']['ama']['threshold_margin']
        self.hl_window = config['buffer']['highlights']['ama']['window']

        self.hl_threshold = self.hl_threshold_start + self.hl_threshold_margin
        self.hl_ep_len_dq = collections.deque(maxlen=self.hl_window )
    
    def store_episode(self,episode):
        if len(episode) <= self.hl_threshold:
            for (o, a, r, o2, d) in episode:
                self.hl_replay_buffer.store(o, a, r, o2, d)

            self.hl_ep_len_dq.append(len(episode))
            if len(self.hl_ep_len_dq) == self.hl_window: self.hl_threshold = np.mean(self.hl_ep_len_dq) + self.hl_threshold_margin
