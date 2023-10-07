import numpy as np
import random
import math

from rltrain.algos.cl.CL import CL


class NoCL(CL):

    def __init__(self, config, env, replay_buffer):
        super(NoCL, self).__init__(config, env, replay_buffer)

        self.cl_ratio = 1
        self.cl_obj_ratio = 1
        self.cl_goal_ratio = 1
   
    def reset_env(self,t):
        self.env.reset()
        return self.env.get_obs()
    




     