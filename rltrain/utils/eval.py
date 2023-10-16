import os
import math
import numpy as np
import torch
import time
import gymnasium as gym
import matplotlib.pyplot as plt

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

    def eval_agent(self,model_name=90,num_display_episode=10, headless=True, time_delay=0.02, current_dir = None, outdir = "eval", figid = "X"):

        # Load model
        path = self.logger.get_model_save_path(model_name)
        #path = os.path.join(current_dir,'logs/0928_A_PandaPush-v3_sac_controldiscrete_const/3/model_backup/model_best_model')
        print(path)

        self.agent.load_weights(path,mode="pi",eval=True)

        # Create Env
        self.config['environment']['headless'] = headless
        self.env = make_env(self.config,self.config_framework)

        # Start Eval
        ep_rets = []
        ep_lens = []
        success_num = 0.0
        for j in tqdm(range(num_display_episode), desc ="Eval: ", leave=True):
            o, d, ep_ret, ep_len = self.env.reset(), False, 0, 0
            self.env.ep_o_start = o.copy()
            
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
            tqdm.write("Len: " + str(ep_len) + " | " + str(info['is_success']))
            if info['is_success'] == True:  success_num += 1        

        # Shutdown environment
        self.env.shuttdown() 

        ep_ret_avg = sum(ep_rets) / len(ep_rets)
        mean_ep_length = sum(ep_lens) / len(ep_lens)
        success_rate = success_num / num_display_episode
        
        print("#########################################")
        print("ep_ret_avg: " + str(ep_ret_avg))
        print("mean_ep_length: " + str(mean_ep_length))
        print("success_rate: " + str(success_rate))
        print("#########################################")

    
    def eval_agent_stats(self,model_name=90,num_display_episode=10, headless=True, time_delay=0.02, current_dir = None, outdir = "eval", figid = "X"):

        # Load model
        path = self.logger.get_model_save_path(model_name)
        #path = os.path.join(current_dir,'logs/0928_A_PandaPush-v3_sac_controldiscrete_const/3/model_backup/model_best_model')
        print(path)

        self.agent.load_weights(path,mode="pi",eval=True)

        # Create Env
        self.config['environment']['headless'] = headless
        self.env = make_env(self.config,self.config_framework)

        # Init heatmap
        heatmap_res = 10
        heatmap_obj = np.zeros((heatmap_res, heatmap_res),dtype=int)
        heatmap_goal = np.zeros((heatmap_res, heatmap_res),dtype=int)  
        bar_dist = np.zeros((heatmap_res),dtype=int) 
        bar_angle = np.zeros((heatmap_res),dtype=int)

        init_ranges = self.env.get_init_ranges()

        print(init_ranges)

        obj_range_low = init_ranges['obj_range_low']
        obj_range_high = init_ranges['obj_range_high']
        goal_range_low = init_ranges['goal_range_low']
        goal_range_high = init_ranges['goal_range_high']
        
        bins_obj = []
        for i in range(2): bins_obj.append(np.linspace(obj_range_low[i], obj_range_high[i], num=heatmap_res+1, retstep=False))
        
        bins_goal = []
        for i in range(2): bins_goal.append(np.linspace(goal_range_low[i], goal_range_high[i], num=heatmap_res+1, retstep=False))
        
        bins_dist = np.linspace(0.0, 0.8, num=heatmap_res+1, retstep=False)
        bins_angle = np.linspace(-90.0, 90.0, num=heatmap_res+1, retstep=False)

        # Start Eval
        ep_rets = []
        ep_lens = []
        success_num = 0.0
        for j in tqdm(range(num_display_episode), desc ="Eval: ", leave=True):
            o, d, ep_ret, ep_len = self.env.reset(), False, 0, 0
            self.env.ep_o_start = o.copy()
            
            obj_init = self.env.get_achieved_goal_from_obs(o)
            goal = self.env.get_desired_goal_from_obs(o)

            obj_dig_x = min(max(np.digitize(obj_init[0], bins_obj[0]) - 1,0),heatmap_res - 1)
            obj_dig_y = min(max(np.digitize(obj_init[1], bins_obj[1]) - 1,0),heatmap_res - 1)
            goal_dig_x = min(max(np.digitize(goal[0], bins_goal[0]) - 1,0),heatmap_res - 1)
            goal_dig_y = min(max(np.digitize(goal[1], bins_goal[1]) - 1,0),heatmap_res - 1)

            distance =  np.linalg.norm(obj_init[:2] - goal[:2], axis=-1)
            distance_dig = min(max(np.digitize(distance, bins_dist) - 1,0),heatmap_res - 1)

            delta_y = goal[1] - obj_init[1]
            delta_x = goal[0] - obj_init[0]
            angle = math.degrees(math.atan(delta_y/delta_x))
            #tqdm.write(str(angle))
            angle_dig = min(max(np.digitize(angle, bins_angle) - 1,0),heatmap_res - 1)

            
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
            tqdm.write("Len: " + str(ep_len) + " | " + str(info['is_success']))
            if info['is_success'] == True: 
                # tqdm.write("O: " + str(self.env.get_achieved_goal_from_obs(o)))
                # tqdm.write("Goal: " + str(self.env.env.task.get_goal()))
                success_num += 1
                heatmap_obj[obj_dig_x][obj_dig_y] += 1
                heatmap_goal[goal_dig_x][goal_dig_y] += 1
                bar_dist[distance_dig] += 1
                bar_angle[angle_dig] += 1
            else:
                heatmap_obj[obj_dig_x][obj_dig_y] -= 1
                heatmap_goal[goal_dig_x][goal_dig_y] -= 1
                bar_dist[distance_dig] -= 1
                bar_angle[angle_dig] -= 1

            # print("--------------------------------")
            # print("ep_ret: " + str(ep_ret) + " | ep_len : " + str(ep_len) + " | Success: " + str(info['is_success']) + " | Goal: " + str(goal))
        

        # Shutdown environment
        self.env.shuttdown() 

        ep_ret_avg = sum(ep_rets) / len(ep_rets)
        mean_ep_length = sum(ep_lens) / len(ep_lens)
        success_rate = success_num / num_display_episode
        
        print("#########################################")
        print("ep_ret_avg: " + str(ep_ret_avg))
        print("mean_ep_length: " + str(mean_ep_length))
        print("success_rate: " + str(success_rate))
        print("#########################################")
        
        fig, ax = plt.subplots()
        im = ax.imshow(heatmap_obj, cmap=plt.cm.RdBu)
        plt.title("Init object position")

        # HEATMAP OBJ
        for i in range(heatmap_res):
            for j in range(heatmap_res):
                text = ax.text(j, i, heatmap_obj[i, j],
                            ha="center", va="center", color="white")

        plt.savefig(os.path.join(current_dir,outdir,figid, figid+"_init_obj_heatmap.png"))
        #plt.show()
        plt.clf()
        plt.cla()

        fig, ax = plt.subplots()
        im = ax.imshow(heatmap_goal, cmap=plt.cm.RdBu)
        plt.title("Goal position")

        # HEATMAP GOAL
        for i in range(heatmap_res):
            for j in range(heatmap_res):
                text = ax.text(j, i, heatmap_goal[i, j],
                            ha="center", va="center", color="white")

        plt.savefig(os.path.join(current_dir,outdir,figid, figid+"_goal_heatmap.png"))
        #plt.show()
        plt.clf()
        plt.cla()

        # BAR DISTANCE
        barlist = plt.bar(bins_dist[:-1], bar_dist, color ='lightskyblue', width = (0.8/(heatmap_res*1.2)))

        for i in range(bar_dist.shape[0]):
            if bar_dist[i] < 0: barlist[i].set_color('lightcoral')
        
        plt.title("Obj - goal distance")
        plt.savefig(os.path.join(current_dir,outdir,figid,figid+"_dist_bar.png"))
        #plt.show()
        plt.clf()
        plt.cla()

        # BAR ANGLE
        barlist = plt.bar(bins_angle[:-1], bar_angle, color ='lightskyblue', width = (180.0/(heatmap_res*1.2)))

        for i in range(bar_angle.shape[0]):
            if bar_angle[i] < 0: barlist[i].set_color('lightcoral')
        
        plt.title("Obj - goal angle")
        plt.savefig(os.path.join(current_dir,outdir,figid,figid+"_angle_bar.png"))
        #plt.show()
        plt.clf()
        plt.cla()
    
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

        for _ in tqdm(range(num_display_episode), desc ="Create video: ", leave=True):

            o_dict, d, ep_ret, ep_len = env.reset(), False, 0, 0
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