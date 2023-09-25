import numpy as np
import collections

from rltrain.algos.cl_teachers.CL import CL

class ExampleByExampleCL(CL):

    def __init__(self, config, env, replay_buffer):
        super(ExampleByExampleCL, self).__init__(config, env, replay_buffer)

        # ExampleByExample
        self.cl_conv_cond = self.config['trainer']['cl']['examplebyexample']['conv_cond']
        self.cl_dequeu_maxlen = config['trainer']['cl']['examplebyexample']['window_size']
        self.cl_ep_success_dq = collections.deque(maxlen=self.cl_dequeu_maxlen)
        self.store_success_rate = True
        self.cl_ratio = 1.0     

        self.same_setup_num = 0
        self.same_setup_num_dq = collections.deque(maxlen=10)
       
    def update_setup(self,t):
        if t == 0:
            self.cl_ratio = 0.0 
            goal_low, goal_high, obj_low, obj_high = self.get_range(self.cl_ratio)
            self.desired_goal = np.random.uniform(goal_low, goal_high)
            self.object_position =  np.random.uniform(obj_low, obj_high)    
            self.cl_ratio = 1.0 
            self.same_setup_num = 0
        elif len(self.cl_ep_success_dq) == self.cl_dequeu_maxlen: 
                success_rate = np.mean(self.cl_ep_success_dq)
                if success_rate > self.cl_conv_cond:
                    goal_low, goal_high, obj_low, obj_high = self.get_range(self.cl_ratio)
                    self.desired_goal = np.random.uniform(goal_low, goal_high)
                    self.object_position =  np.random.uniform(obj_low, obj_high)
                    self.cl_ep_success_dq.clear()
                    self.same_setup_num_dq.append(self.same_setup_num)
                    self.same_setup_num = 0
                
        self.same_setup_num += 1
    
    def reset_env(self,t):

        self.update_setup(t)
        self.env.reset()
        self.env.load_state(robot_joints= None, desired_goal = self.desired_goal, object_position = self.object_position)
        
        return self.env.get_obs()
    
    def clear_cl_ep_success_dq(self):
        for _ in range(self.cl_ep_success_dq.maxlen):
            self.cl_ep_success_dq.append(0.0)
        
    




     