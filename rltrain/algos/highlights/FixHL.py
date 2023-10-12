from rltrain.algos.highlights.HL import HL

class FixHL(HL):
    def __init__(self, config):
        super(FixHL, self).__init__(config)

        self.hl_threshold = config['buffer']['highlights']['fix']['threshold']
    
    
    def store_episode(self,episode):
        if len(episode) <= self.hl_threshold:
            for (o, a, r, o2, d) in episode:
                self.hl_replay_buffer.store(o, a, r, o2, d)
