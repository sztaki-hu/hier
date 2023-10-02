import os
import math
import numpy as np
import torch
import time
import gymnasium as gym

from tqdm import tqdm
from numpngw import write_apng  # pip install numpngw
import cv2

from rltrain.envs.builder import make_env

class Eval:

    def __init__(self,agent,logger,config,config_framework):
        #self.env = env
        self.agent = agent
        self.logger = logger
        self.config = config
        self.config_framework = config_framework

        self.max_ep_len = config['sampler']['max_ep_len']  
        self.agent_type = config['agent']['type']  

    
    def eval_agent(self,model_name=90,num_display_episode=10, headless=True, time_delay=0.02, current_dir = None):

        # Load model
        path = self.logger.get_model_save_path(model_name)
        #path = os.path.join(current_dir,'logs/0928_A_PandaPush-v3_sac_controldiscrete_const/3/model_backup/model_best_model')

        self.agent.load_weights(path,mode="pi",eval=True)

        # Create Env
        self.config['environment']['headless'] = headless
        self.env = make_env(self.config,self.config_framework)

        # Start Eval
        ep_rets = []
        ep_lens = []
        success_num = 0.0
        for j in range(num_display_episode):
            [o, info], d, ep_ret, ep_len = self.env.reset_with_init_check(), False, 0, 0
            self.env.ep_o_start = o.copy()
            goal = self.env.get_desired_goal_from_obs(o)
            while not(d or (ep_len == self.max_ep_len)):
                # Take deterministic actions at test time 
                if self.agent_type == 'sac':
                    a = self.agent.get_action(o, True)
                elif self.agent_type == 'td3':
                    a = self.agent.get_action(o, 0)
                o, r, terminated, truncated, info = self.env.step(a)
                d = terminated or truncated
                ep_ret += r
                ep_len += 1
                if headless == False: time.sleep(time_delay)
            ep_rets.append(ep_ret)
            ep_lens.append(ep_len)
            if info['is_success'] == True: success_num += 1
            print("--------------------------------")
            print("ep_ret: " + str(ep_ret) + " | ep_len : " + str(ep_len) + " | Success: " + str(info['is_success']) + " | Goal: " + str(goal))
        
        ep_ret_avg = sum(ep_rets) / len(ep_rets)
        mean_ep_length = sum(ep_lens) / len(ep_lens)
        success_rate = success_num / num_display_episode
        
        print("#########################################")
        print("ep_ret_avg: " + str(ep_ret_avg))
        print("mean_ep_length: " + str(mean_ep_length))
        print("success_rate: " + str(success_rate))
        print("#########################################")
        
        # Shutdown environment
        self.env.shuttdown() 
    
    def save_video(self,model_name="best_model",num_display_episode=10, current_dir = None, outdir = "vids", save_name = "video.png"):

        # Load model
        path = self.logger.get_model_save_path(model_name)

        self.agent.load_weights(path,mode="pi",eval=True)

        # Create Env
        self.config['environment']['headless'] = True

        assert self.config['environment']['name'] == "gympanda"
        env = gym.make(self.config['environment']['task']['name'], render_mode="rgb_array")

        # Start Eval
        ep_rets = []
        ep_lens = []
        images = []
        success_list = []

        for _ in tqdm(range(num_display_episode), desc ="Training: ", leave=True):

            [o_dict, info], d, ep_ret, ep_len = env.reset(), False, 0, 0
            o = np.concatenate((o_dict['observation'], o_dict['desired_goal']))
            images.append(env.render())
            

            while not(d or (ep_len == self.max_ep_len)):
                # Take deterministic actions at test time 
                if self.agent_type == 'sac':
                    a = self.agent.get_action(o, True)
                elif self.agent_type == 'td3':
                    a = self.agent.get_action(o, 0)
                o_dict, r, terminated, truncated, info  = env.step(a)
                o = np.concatenate((o_dict['observation'], o_dict['desired_goal']))

                images.append(self.draw_results(env.render(),success_list))
                d = terminated or truncated
                ep_ret += r
                ep_len += 1
            
            ep_rets.append(ep_ret)
            ep_lens.append(ep_len)

            if info['is_success'] == True: 
                success_list.append(1.0)
                frame_color = (0,255,0)
            else: 
                success_list.append(0.0)
                frame_color = (0,0,255)

            img_np = env.render()
            images.append(img_np.copy())
  
            # Draw frame 
            img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            img_cv = cv2.rectangle(img_cv, (0, 0), (img_np.shape[1],img_np.shape[0]), frame_color, 5)
            for _ in range(10): images.append(self.draw_results(img_cv.copy(),success_list))

            tqdm.write("ep_ret: " + str(ep_ret) + " | ep_len : " + str(ep_len) + " | Success: " + str(info['is_success']))
        
        # Shutdown environment
        env.close()

        write_apng(os.path.join(current_dir,outdir,save_name), images, delay=40)

        ep_ret_avg = sum(ep_rets) / len(ep_rets)
        mean_ep_length = sum(ep_lens) / len(ep_lens)
        success_rate = sum(success_list) / len(success_list)
        
        print("#########################################")
        print("ep_ret_avg: " + str(ep_ret_avg))
        print("mean_ep_length: " + str(mean_ep_length))
        print("success_rate: " + str(success_rate))
        print("#########################################")
    
    def draw_results(self,img_np,success_list):

        img_shape = img_np.shape

        # NP2CV
        img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # Draw history
        delta = 20
        margin = 10
        
        for i in range (len(success_list)):

            start_point = ((i+1) * delta, img_shape[0] - margin - delta)
            end_point = ((i+2) * delta, img_shape[0] - margin)

            ep_color = (0,255,0) if success_list[i] == 1.0 else (255,0,0)
            img_cv = cv2.rectangle(img_cv, start_point, end_point, ep_color, -1)
            img_cv = cv2.rectangle(img_cv, start_point, end_point, (0,0,0), 1)
        
        return img_cv