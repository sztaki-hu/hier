import numpy as np
import random
import math

from rltrain.algos.cl_teachers.CL import CL


class NoCL(CL):
   
    def reset_env(self,t):
        self.env.reset()
        return self.env.get_obs()
    




     