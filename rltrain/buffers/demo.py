import numpy as np
import time
import random
import math
from tqdm import tqdm
from pyquaternion import Quaternion

from rltrain.buffers.replay import ReplayBuffer
from rltrain.envs.builder import make_env

DEMO_GENERATE_TYPE_LIST = ['normal','subgoal_attention']

class Demo:
    def __init__(self,logger,config):
        self.logger = logger
        self.config = config  
            

        self.demo_use = config['demo']['demo_use']  
        self.demo_buffer_size = int(config['demo']['demo_buffer_size'])  
        self.demo_name = config['demo']['demo_name'] 
        self.demo_block_order = config['demo']['demo_block_order']
        self.demo_change_nstep = config['demo']['demo_change_nstep']  
        self.demo_generate_type = config['demo']['demo_generate_type'] 
        self.demo_generate_params = config['demo']['demo_generate_params']
        
        assert self.demo_generate_type in DEMO_GENERATE_TYPE_LIST

        self.reward_scalor = config['environment']['reward']['reward_scalor']

        self.obs_dim = config['environment']['obs_dim']
        self.act_dim = config['environment']['act_dim']
        self.action_space = config['agent']['action_space']
        self.gamma = config['agent']['gamma'] 
        self.n_step = config['agent']['n_step'] 
        self.batch_size = config['trainer']['batch_size'] 

        self.task_name = self.config['environment']['task']['name']
        self.task_params = self.config['environment']['task']['params']
        if len(self.task_params) > 0: self.target_blocks_num = int(self.task_params[0])
        if len(self.task_params) > 1: self.distractor_blocks_num = int(self.task_params[1])
        if len(self.task_params) > 2: self.tower_height = int(self.task_params[2])

        self.max_ep_len = int(self.config["sampler"]["max_ep_len"])
        self.env_name = config['environment']['name']
        self.env_headless = self.config['environment']['headless']

        self.demo_buffer = ReplayBuffer(obs_dim=self.obs_dim, act_dim=self.act_dim, size=self.demo_buffer_size)

    def create_demos(self):  

        if self.task_name == "reach_target_no_distractors":
            if self.action_space == "joint":
                self.create_demo_reach_joint()
        if self.task_name == "stack_blocks":
            if (self.action_space == "pick_and_place_2d") or (self.action_space == "pick_and_place_3d"):
                self.create_demos_stack_blocks_pick_and_place()
            elif self.action_space == "pick_and_place_3d_quat":
                self.create_demos_stack_blocks_pick_and_place_3d_quat()
            elif self.action_space == "pick_and_place_3d_z90":        
                self.create_demos_stack_blocks_pick_and_place_3d_z90()
        elif self.task_name == "MountainCarContinuous-v0":
            self.create_demo_MountainCarContinuous()
    
    def create_demo_reach_joint(self):

        self.env = make_env(self.config)

        pbar = tqdm(total=int(self.demo_buffer_size),colour="green")
        ep_num = 0
        unsuccessful_num = 0
        while ep_num<int(self.demo_buffer_size):

            ep_transitions = []
            ret = 0

            ## Reset Env
            while True:

                try:
                    o = self.env.reset_once()
                    if self.env.init_state_valid():
                        break
                    else:
                        tqdm.write('Init state is not valid. Repeat env reset.')
                        time.sleep(0.1)
                except:
                    tqdm.write('Could not reset the environment. Repeat env reset.')
                    time.sleep(1)
            

            d = 0

            for _ in range(self.max_ep_len):

                a = np.random.uniform(-1.0,1.0,self.act_dim)
                if self.action_space == "jointgripper":
                    a[6] = 1.0 if a[6] > 0.5 else 0.0
                #print(a)
                
                try:
                    o2, r, d, info = self.env.step(a)
                    ep_transitions.append((o, a, r, o2, d))
                    o = o2
                    if d == 1:
                        break
                except:
                    tqdm.write("Error in simulation, this demonstration is not added")
                    d = 0
                    break

            if d == 1:   
                self.demo_buffer.store_episode_nstep(ep_transitions,self.n_step,self.gamma)
                ep_num+=1
                pbar.update(1)
                
            else:
                unsuccessful_num += 1   
                tqdm.write("The demonstration is not successful, thus it is not added | Num: " + str(unsuccessful_num) + " | Return: " + str(ret) +  " | Obs: " + str(o))    
        
        pbar.close()
        
        self.env.shuttdown()

        self.logger.save_demos(self.demo_name,  self.demo_buffer)

    def create_demo_MountainCarContinuous(self):

        self.env = make_env(self.config)

        pbar = tqdm(total=int(self.demo_buffer_size),colour="green")
        t = 0
        unsuccessful_num = 0
        while t<int(self.demo_buffer_size):

            ep_transitions = []
            ret = 0

            ## Reset Env
            while True:

                try:
                    o = self.env.reset_once()
                    if self.env.init_state_valid():
                        break
                    else:
                        tqdm.write('Init state is not valid. Repeat env reset.')
                        time.sleep(0.1)
                except:
                    tqdm.write('Could not reset the environment. Repeat env reset.')
                    time.sleep(1)
            
            a = np.array([1.0])
            d = 0
            for _ in range(self.max_ep_len):

                if self.env_name == "gym" and self.env_headless == False:
                    self.env.render()
                    #time.sleep(0.1)

                # if math.isclose(o[1], 0, abs_tol = 0.001):
                #     a[0] = -1.0 if o[0] > 0 else +1.0
                
                if o[1] < 0 and a[0] > 0:
                    a[0] = -1.0
                elif o[1] > 0 and a[0] < 0:
                    a[0] = +1.0

                #tqdm.write("a: " + str(a))
                # o2, r, d, info = self.env.step(a)
                # o = o2
                try:
                    o2, r, d, info = self.env.step(a)
                    ep_transitions.append((o, a, r, o2, d))
                    o = o2
                    if d == 1:
                        break
                except:
                    tqdm.write("Error in simulation, this demonstration is not added")
                    d = 0
                    break

            if d == 1:   
                self.demo_buffer.store_episode_nstep(ep_transitions,self.n_step,self.gamma)
                t+=len(ep_transitions)
                pbar.update(len(ep_transitions))
                
            else:
                unsuccessful_num += 1   
                tqdm.write("The demonstration is not successful, thus it is not added | Num: " + str(unsuccessful_num) + " | Return: " + str(ret) +  " | Obs: " + str(o))    
        
        pbar.close()
        
        self.env.shuttdown()

        self.logger.save_demos(self.demo_name,  self.demo_buffer)

    def create_demos_xyz(self):  

        self.env = make_env(self.config)

        for _ in tqdm(range(self.demo_buffer_size), desc ="Loading demos: ", colour="green"):  
            o = self.env.reset()
            a = o
            try:
                o2, r, d, info = self.env.step(a)
                if r > 0:
                     self.demo_buffer.store(o, a, r, o2, d)
                else:
                    tqdm.write("The demonstration is not successful, thus it is not added") 
            except:
                tqdm.write("Error in simulation, this demonstration is not added")         
        self.logger.save_demos(self.demo_name,  self.demo_buffer)

        self.env.shuttdown()

        # batch = self.demo_buffer.sample_batch(self.batch_size) 
        # print(batch)
        # print(self.demo_buffer.ptr)
        
    def create_demos_stack_blocks_pick_and_place_3d_z90(self):  

        self.env = make_env(self.config)
        unsuccessful_num = 0

        assert self.target_blocks_num >= self.tower_height

        if self.demo_generate_type == 'subgoal_attention':     
            subgoal_add_th = []    
            last_th = 0.0
            for i in range(self.tower_height):
                subgoal_add_th.append(float(self.demo_generate_params[i])+last_th)
                last_th = subgoal_add_th[-1]
            subgoal_ratio = np.zeros(self.tower_height)

        pbar = tqdm(total=int(self.demo_buffer_size),colour="green")
        t = 0
        while t<int(self.demo_buffer_size):
            
            ## Reset Env
            while True:

                try:
                    o = self.env.reset_once()
                    if self.env.init_state_valid():
                        break
                    else:
                        tqdm.write('Init state is not valid. Repeat env reset.')
                        time.sleep(0.1)
                except:
                    tqdm.write('Could not reset the environment. Repeat env reset.')
                    time.sleep(1)
            
            

            ## Compute order
            target_index =  (0, 1, 2, 3)
            target = o[[target_index[0],target_index[1],target_index[2],target_index[3]]]

            blocks_init = []
            dists = []
            for j in range(1,self.target_blocks_num+1):
                block_index =  (j * 4, j * 4 + 1, j * 4 + 2,j * 4 + 3)
                block = o[[block_index[0],block_index[1],block_index[2],block_index[3]]]
                dists.append(np.sum(np.square(target - block)))
                blocks_init.append(block)         
            
            if self.demo_block_order == "dist_based":
                blocks = []
                for _ in range(self.target_blocks_num):
                    index = np.argmin(dists)
                    blocks.append(blocks_init[index])
                    dists[index] = float('inf')
            elif self.demo_block_order == "random":
                print("Not implemented yet")
            elif self.demo_block_order == "as_init":
                blocks = blocks_init

            ## Execute demo
            ep_transitions = []
            ret = 0
            
            for i in range(self.tower_height):
                
                a = np.hstack((blocks[i],target)) 
                a[6] += 0.01 + 0.03 * i

                o2, r, d, info = self.env.step(a)
                ep_transitions.append((o, a, r, o2, d))
                o = o2
                ret += r

                # try:
                #     o2, r, d, info = self.env.step(a)
                #     ep_transitions.append((o, a, r, o2, d))
                #     o = o2
                #     ret += r
                # except:
                #     tqdm.write("Error in simulation, this demonstration is not added")
                #     ret = -1
                #     break

            if ret >= self.reward_scalor:
                if self.demo_generate_type == 'normal':
                    self.demo_buffer.store_episode_nstep(ep_transitions,self.n_step,self.gamma)
                    t+=len(ep_transitions)
                    pbar.update(len(ep_transitions))
                elif self.demo_generate_type == 'subgoal_attention':
                    randnum = random.uniform(0.0, 1.0)
                    for j in range(self.tower_height):
                        if randnum < subgoal_add_th[j]:
                            self.demo_buffer.store_episode_nstep(ep_transitions[j:],self.n_step,self.gamma)
                            subgoal_ratio[j] += 1
                            break
                    increment = len(ep_transitions[j:])
                    t+=increment
                    pbar.update(increment)
                    

            else:
                unsuccessful_num += 1   
                tqdm.write("The demonstration is not successful, thus it is not added | Num: " + str(unsuccessful_num) + " | Return: " + str(ret) +  " | Obs: " + str(o))    
        
        pbar.close()
        
        self.env.shuttdown()

        if self.demo_generate_type == 'subgoal_attention':
            subgoal_ratio = subgoal_ratio / np.sum(subgoal_ratio)
            print("Subgoal ratio: " + str(subgoal_ratio))

        self.logger.save_demos(self.demo_name,  self.demo_buffer)

        # batch = self.demo_buffer.sample_batch(self.batch_size) 
        # print(batch)
        # print(self.demo_buffer.ptr)

    def create_demos_stack_blocks_pick_and_place_3d_quat(self):  

        self.env = make_env(self.config)
        unsuccessful_num = 0

        assert self.target_blocks_num >= self.tower_height

        if self.demo_generate_type == 'subgoal_attention':     
            subgoal_add_th = []    
            last_th = 0.0
            for i in range(self.tower_height):
                subgoal_add_th.append(float(self.demo_generate_params[i])+last_th)
                last_th = subgoal_add_th[-1]
            subgoal_ratio = np.zeros(self.tower_height)

        pbar = tqdm(total=int(self.demo_buffer_size),colour="green")
        t = 0
        while t<int(self.demo_buffer_size):
            
            ## Reset Env
            while True:
                
                try:
                    o = self.env.reset_once()
                    if self.env.init_state_valid():
                        break
                    else:
                        tqdm.write('Init state is not valid. Repeat env reset.')
                        time.sleep(0.1)
                except:
                    tqdm.write('Could not reset the environment. Repeat env reset.')
                    time.sleep(1)

            ## Compute order
            target_index =  (0, 1, 2, 3, 4, 5, 6)
            target = o[[target_index[0],target_index[1],target_index[2],target_index[3],target_index[4],target_index[5],target_index[6]]]

            blocks_init = []
            dists = []
            for j in range(1,self.target_blocks_num+1):
                block_index =  (j * 7, j * 7 + 1, j * 7 + 2,j * 7 + 3,j * 7 + 4,j * 7 + 5,j * 7 + 6)
                block = o[[block_index[0],block_index[1],block_index[2],block_index[3],block_index[4],block_index[5],block_index[6]]]
                dists.append(np.sum(np.square(target - block)))
                blocks_init.append(block)         
            
            if self.demo_block_order == "dist_based":
                blocks = []
                for _ in range(self.target_blocks_num):
                    index = np.argmin(dists)
                    blocks.append(blocks_init[index])
                    dists[index] = float('inf')
            elif self.demo_block_order == "random":
                print("Not implemented yet")
            elif self.demo_block_order == "as_init":
                blocks = blocks_init

            ## Execute demo
            ep_transitions = []
            ret = 0
            
            for i in range(self.tower_height):
                
                if self.action_space == "pick_and_place_2d":
                    a = np.array([blocks[i][0],blocks[i][1],target[0],target[1]])
                elif self.action_space == "pick_and_place_3d":
                    a = np.hstack((blocks[i],target)) 
                    a[5] += 0.01 + 0.03 * i
                elif self.action_space == "pick_and_place_3d_quat":
                    a = np.hstack((blocks[i],target)) 
                    a[9] += 0.01 + 0.03 * i

                    q = self.rlbench2pyquat(a[3:7]) # (x,y,z,w) --> (w,x,y,z)
                    y_180 = Quaternion(axis=[0, 1, 0], angle=3.14159265) # Rotate 180 about Y
                    q2 = q * y_180
                    a[3:7] = self.pyquat2rlbench(q2) # (w,x,y,z) --> (x,y,z,w)  

                    q = self.rlbench2pyquat(a[10:14]) # (x,y,z,w) --> (w,x,y,z)
                    y_180 = Quaternion(axis=[0, 1, 0], angle=3.14159265) # Rotate 180 about Y
                    q2 = q * y_180
                    a[10:14] = self.pyquat2rlbench(q2) # (w,x,y,z) --> (x,y,z,w)  

                o2, r, d, info = self.env.step(a)
                ep_transitions.append((o, a, r, o2, d))
                o = o2
                ret += r

                # try:
                #     o2, r, d, info = self.env.step(a)
                #     ep_transitions.append((o, a, r, o2, d))
                #     o = o2
                #     ret += r
                # except:
                #     tqdm.write("Error in simulation, this demonstration is not added")
                #     ret = -1
                #     break

            if ret >= self.reward_scalor:
                if self.demo_generate_type == 'normal':
                    self.demo_buffer.store_episode_nstep(ep_transitions,self.n_step,self.gamma)
                    t+=len(ep_transitions)
                    pbar.update(len(ep_transitions))
                elif self.demo_generate_type == 'subgoal_attention':
                    randnum = random.uniform(0.0, 1.0)
                    for j in range(self.tower_height):
                        if randnum < subgoal_add_th[j]:
                            self.demo_buffer.store_episode_nstep(ep_transitions[j:],self.n_step,self.gamma)
                            subgoal_ratio[j] += 1
                            break
                    increment = len(ep_transitions[j:])
                    t+=increment
                    pbar.update(increment)
                    

            else:
                unsuccessful_num += 1   
                tqdm.write("The demonstration is not successful, thus it is not added | Num: " + str(unsuccessful_num) + " | Return: " + str(ret) +  " | Obs: " + str(o))    
        
        pbar.close()
        
        self.env.shuttdown()

        if self.demo_generate_type == 'subgoal_attention':
            subgoal_ratio = subgoal_ratio / np.sum(subgoal_ratio)
            print("Subgoal ratio: " + str(subgoal_ratio))

        self.logger.save_demos(self.demo_name,  self.demo_buffer)

        # batch = self.demo_buffer.sample_batch(self.batch_size) 
        # print(batch)
        # print(self.demo_buffer.ptr)

    def create_demos_stack_blocks_pick_and_place(self):  

        self.env = make_env(self.config)
        unsuccessful_num = 0

        assert self.target_blocks_num >= self.tower_height

        if self.demo_generate_type == 'subgoal_attention':     
            subgoal_add_th = []    
            last_th = 0.0
            for i in range(self.tower_height):
                subgoal_add_th.append(float(self.demo_generate_params[i])+last_th)
                last_th = subgoal_add_th[-1]
            subgoal_ratio = np.zeros(self.tower_height)

        pbar = tqdm(total=int(self.demo_buffer_size),colour="green")
        t = 0
        while t<int(self.demo_buffer_size):

            ## Reset Env
            while True:
                try:
                    o = self.env.reset_once()
                    if self.env.init_state_valid():
                        break
                    else:
                        tqdm.write('Init state is not valid. Repeat env reset.')
                        time.sleep(0.1)
                except:
                    tqdm.write('Could not reset the environment. Repeat env reset.')
                    time.sleep(1)

            ## Compute order
            target_index =  (0, 1, 2)
            target = o[[target_index[0],target_index[1],target_index[2]]]

            blocks_init = []
            dists = []
            for j in range(1,self.target_blocks_num+1):
                block_index =  (j * 3, j * 3 + 1, j * 3 + 2)
                block = o[[block_index[0],block_index[1],block_index[2]]]
                dists.append(np.sum(np.square(target - block)))
                blocks_init.append(block)         
            
            if self.demo_block_order == "dist_based":
                blocks = []
                for _ in range(self.target_blocks_num):
                    index = np.argmin(dists)
                    blocks.append(blocks_init[index])
                    dists[index] = float('inf')
            elif self.demo_block_order == "random":
                print("Not implemented yet")
            elif self.demo_block_order == "as_init":
                blocks = blocks_init

            ## Execute demo
            ep_transitions = []
            ret = 0
            
            for i in range(self.tower_height):
                
                if self.action_space == "pick_and_place_2d":
                    a = np.array([blocks[i][0],blocks[i][1],target[0],target[1]])
                elif self.action_space == "pick_and_place_3d":
                    a = np.hstack((blocks[i],target)) 
                    a[5] += 0.01 + 0.03 * i

                o2, r, d, info = self.env.step(a)
                ep_transitions.append((o, a, r, o2, d))
                o = o2
                ret += r

                # try:
                #     o2, r, d, info = self.env.step(a)
                #     ep_transitions.append((o, a, r, o2, d))
                #     o = o2
                #     ret += r
                # except:
                #     tqdm.write("Error in simulation, this demonstration is not added")
                #     ret = -1
                #     break

            if ret >= self.reward_scalor:
                if self.demo_generate_type == 'normal':
                    self.demo_buffer.store_episode_nstep(ep_transitions,self.n_step,self.gamma)
                    t+=len(ep_transitions)
                    pbar.update(len(ep_transitions))
                elif self.demo_generate_type == 'subgoal_attention':
                    randnum = random.uniform(0.0, 1.0)
                    for j in range(self.tower_height):
                        if randnum < subgoal_add_th[j]:
                            self.demo_buffer.store_episode_nstep(ep_transitions[j:],self.n_step,self.gamma)
                            subgoal_ratio[j] += 1
                            break
                    increment = len(ep_transitions[j:])
                    t+=increment
                    pbar.update(increment)
                    

            else:
                unsuccessful_num += 1   
                tqdm.write("The demonstration is not successful, thus it is not added | Num: " + str(unsuccessful_num) + " | Return: " + str(ret) +  " | Obs: " + str(o))    
        
        pbar.close()
        
        self.env.shuttdown()

        if self.demo_generate_type == 'subgoal_attention':
            subgoal_ratio = subgoal_ratio / np.sum(subgoal_ratio)
            print("Subgoal ratio: " + str(subgoal_ratio))

        self.logger.save_demos(self.demo_name,  self.demo_buffer)

        # batch = self.demo_buffer.sample_batch(self.batch_size) 
        # print(batch)
        # print(self.demo_buffer.ptr)
    
    def clean_up_old_demo(self):
        if self.demo_exists():
            self.logger.remove_old_demo(self.demo_name)
            print("Old demo removed")
            

    def demo_exists(self):
        return self.logger.demo_exists(self.demo_name)

    def load_demos(self):
        if self.demo_change_nstep == False:
            return self.logger.load_demos(self.demo_name)
        else:
            return self.load_and_change_nstep()

    def load_and_change_nstep(self):

        base_demo_buffer = self.logger.load_demos(self.demo_name)
        data = base_demo_buffer.get_all()

        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']
        #r_nstep, o_nstep, d_nstep,n_nstep = data['rew_nstep'], data['obs_nstep'], data['done_nstep'], data['n_nstep']

        o = o.detach().numpy()
        a = a.detach().numpy()
        r = r.detach().numpy()
        o2 = o2.detach().numpy()
        d = d.detach().numpy()

        ep_transitions = []
        for i in tqdm(range(d.shape[0]), desc ="Demo changing nstep: ", colour="green"):  
            ep_transitions.append((o[i], a[i], r[i], o2[i], d[i]))
            if d[i] == 1:
                self.demo_buffer.store_episode_nstep(ep_transitions,self.n_step,self.gamma)
                ep_transitions = []
        
        return self.demo_buffer
    
    def pyquat2rlbench(self,quat): # (w,x,y,z) --> (x,y,z,w)
        return np.array([quat[1], quat[2], quat[3], quat[0]])
    def rlbench2pyquat(self,quat): # (x,y,z,w) --> (w,x,y,z)
        return Quaternion(quat[3], quat[0], quat[1], quat[2])
