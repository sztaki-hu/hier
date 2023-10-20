import math

from rltrain.algos.highlights.HL import HL

PACING_PROFILES = ['linear','sqrt','quad']

class PredefinedHL(HL):
    def __init__(self, config):
        super(PredefinedHL, self).__init__(config)

        self.hl_threshold_pacing_profile = config['buffer']['highlights']['predefined']['threshold_pacing_profile']
        assert self.hl_threshold_pacing_profile in PACING_PROFILES

        self.hl_threshold_pacing_sat = config['buffer']['highlights']['predefined']['threshold_pacing_sat']
        assert 0.0 <= self.hl_threshold_pacing_sat <= 1.0
        
        self.hl_threshold_start = config['buffer']['highlights']['predefined']['threshold_start']
        self.hl_threshold_end = config['buffer']['highlights']['predefined']['threshold_end']
        self.hl_threshold = self.hl_threshold_start
        self.hl_threshold_range = self.hl_threshold_end - self.hl_threshold_start
        assert self.hl_threshold_range > 0 

        self.total_timesteps = float(config['trainer']['total_timesteps'])

        
    
    def store_episode(self,episode,info_success,t):

        if self.hl_success_cond and info_success == False: return

        self.update_threshold(t)

        sum_rew = 0
        for (o, a, r, o2, d) in episode:
            sum_rew += r

        if sum_rew >= self.hl_threshold:
            for (o, a, r, o2, d) in episode:
                self.hl_replay_buffer.store(o, a, r, o2, d)
    

    def update_threshold(self,t):
        t = float(t) / self.total_timesteps
        if self.hl_threshold_pacing_profile == "linear":
            hl_threshold_ratio = min(1.0, t / self.hl_threshold_pacing_sat)
        elif self.hl_threshold_pacing_profile == "sqrt":
            hl_threshold_ratio = min(1.0, math.sqrt(t / self.hl_threshold_pacing_sat))
        elif self.hl_threshold_pacing_profile == "quad":
            hl_threshold_ratio = min(1.0, math.pow(t / self.hl_threshold_pacing_sat,2))
        
        print(self.hl_threshold_range)
        print(hl_threshold_ratio)

        self.hl_threshold = self.hl_threshold_start + (self.hl_threshold_range * hl_threshold_ratio)

      

    
    
