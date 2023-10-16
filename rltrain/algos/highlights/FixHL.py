from rltrain.algos.highlights.HL import HL

class FixHL(HL):
    def __init__(self, config):
        super(FixHL, self).__init__(config)

        self.hl_threshold = config['buffer']['highlights']['fix']['threshold']
    
    
    def store_episode(self,episode,info_success):

        if self.hl_success_cond and info_success == False: return

        sum_rew = 0
        for (o, a, r, o2, d) in episode:
            sum_rew += r

        if sum_rew >= self.hl_threshold:
            for (o, a, r, o2, d) in episode:
                self.hl_replay_buffer.store(o, a, r, o2, d)
