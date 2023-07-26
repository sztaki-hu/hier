import numpy as np
import gymnasium as gym
import time

class Env:
    def __init__(self):
        self.env = None
        pass
    
    def reset(self):
        o, _ = self.env.reset()
        return o       
    
    def shuttdown(self):
        self.reset()
        self.env.close()
    
    def init_state_valid(self, o):
        return True  

    def reset_with_init_check(self):
        init_invalid_num = 0
        reset_num = 0
        ## Reset Env
        while True:
            o = self.reset()
            try:
                o = self.reset()
                reset_num += 1
                if self.init_state_valid(o):
                    info = {}
                    info['init_invalid_num'] = init_invalid_num
                    info['reset_num'] = reset_num
                    return o, info
                else:
                    init_invalid_num+=0                
            except:        
                time.sleep(0.1) 

    def is_success(self):
        return False       

    def step(self,action):

        o, r, terminated, truncated, info = self.env.step(action)

        info['is_success'] = True if self.is_success() == True else False

        return o, r, terminated, truncated, info
    
    def random_sample(self):
        return self.env.action_space.sample() 

    def get_max_return(self):
        return None
    

    
  
                 

