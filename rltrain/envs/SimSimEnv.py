import numpy as np
import time

class SimSimEnv:
    def __init__(self,config):
        self.config = config
        self.reward_shaping_use = config['environment']['reward']['reward_shaping_use']
        self.reward_shaping_type = config['environment']['reward']['reward_shaping_type']
        self.obs_dim = config['environment']['obs_dim']
        self.boundary_min = np.array(config['agent']['boundary_min'])[:self.obs_dim]
        self.boundary_max = np.array(config['agent']['boundary_max'])[:self.obs_dim]
        self.target_position = np.zeros(self.obs_dim)
        self.reset()

    def shuttdown(self):
        return None
    
    def reset(self):
        self.target_position = np.random.uniform(low=self.boundary_min, high=self.boundary_max, size=(self.obs_dim))
        return self.target_position
    
    def step(self,a):
        d = np.allclose(a[:self.obs_dim], self.target_position, rtol=0.0, atol=0.02, equal_nan=False)
        r = float(d) * 100
        info = None
        if self.reward_shaping_use:
            if self.reward_shaping_type == 'mse':
                r = self.reward_shaping_mse(o)
        # avg = (self.boundary_min[0] + self.boundary_max[0]) / 2.0
        # range = abs((self.boundary_max[0] - self.boundary_min[0]))
        # o = (o - avg) / range
        o = self.target_position
        time.sleep(0.05)
        return o, r, d, info
    
    def reward_shaping_mse(self,o):
        return -((o - self.target_position)**2).sum()
    
    def get_target(self): #for demos
        return self.target_position
        


