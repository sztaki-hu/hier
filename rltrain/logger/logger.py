import os
import logging
import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

import pandas as pd


class Logger:
    # init method or constructor
    def __init__(self, current_dir, main_args, light_mode = False):

        self.current_dir = current_dir
        self.trainid = str(main_args.trainid)

        print(main_args)

        if main_args.restart == False:       
            self.config_path = os.path.join(current_dir,main_args.configfile)
        else:
            self.config_path = os.path.join(current_dir,main_args.configfile,self.trainid,"config.yaml")
        
        print(current_dir)
        print(self.config_path)
        
        self.config = self.load_yaml(self.config_path)

        self.logdir = self.config['general']['logdir']
        self.logname = self.config['general']['exp_name'] + "_" + self.config['environment']['task']['name'] + "_" + self.config['agent']['type'] 

        self.demodir = self.config['general']['demodir']
        
        
        log_file_path = os.path.join(self.current_dir,self.logdir, self.logname,self.trainid,'logs.log')
        if os.path.isfile(log_file_path):
            os.remove(log_file_path) 
        
        self.create_folder(os.path.join(self.current_dir,self.logdir, self.logname,self.trainid))

        logging.basicConfig(filename=log_file_path,level=logging.DEBUG)
        self.pylogger = logging.getLogger('mylogger')

        cfg_rlbench = {'path' : self.config_path}
        self.create_folder(os.path.join(self.current_dir, "cfg_rlbench"))

        self.handle_ancient_versions()

        self.check_config_values()

        if main_args.restart == False: 
            self.compute_and_replace_auto_values()

        self.save_yaml(os.path.join(self.current_dir, "cfg_rlbench" ,"config.yaml"),cfg_rlbench)

        # cfg_rlbench_2 = self.load_yaml(os.path.join(self.current_dir, "cfg_rlbench" ,"config.yaml"))
        # print(cfg_rlbench_2)
        # print(cfg_rlbench_2['path'])

        self.create_folder(os.path.join(self.current_dir,self.logdir, self.logname,self.trainid)) 

        self.heatmap_bool = self.config['logger']['heatmap']['bool']
        self.agent_num = int(self.config['agent']['agent_num'])
        if self.agent_num > 1: assert self.config['general']['sync'] == True

        if light_mode == False:
            if main_args.restart == False:     
                for agent_id in range(self.agent_num):
                    self.create_folder(os.path.join(self.current_dir,self.logdir, self.logname,self.trainid,"model_backup",str(agent_id)))
                #self.create_folder(os.path.join(self.current_dir,self.logdir, self.logname,self.trainid,"replay_buffer_backup"))
                #self.create_folder(os.path.join(self.current_dir, self.logdir, self.logname,self.trainid,"plots_raw_data"))
                self.save_yaml(os.path.join(self.current_dir, self.logdir,self.logname,self.trainid,"config.yaml"),self.config)

            train_ret_list = []
            train_ep_len_list = []
            train_pi_loss_list = []
            train_q_loss_list = []
            train_success_list = []
            test_ret_list = []
            test_success_list = []
            for i in range(self.agent_num):
                train_ret_list.append("train/train_ret_"+str(i))
                train_ep_len_list.append("train/train_ep_len_"+str(i))
                train_pi_loss_list.append("train/loss_pi_"+str(i))
                train_q_loss_list.append("train/loss_q_"+str(i))
                train_success_list.append("train/train_success_"+str(i))
                test_ret_list.append("test/test_ret_"+str(i))
                test_success_list.append("test/test_success_"+str(i))
          
            layout = {
                "agents train": {
                    "train_ret": ["Multiline", train_ret_list],
                    "train_ep_len": ["Multiline", train_ep_len_list],
                    "train_pi_loss": ["Multiline", train_pi_loss_list],
                    "train_q_loss": ["Multiline", train_q_loss_list],     
                    "train_success": ["Multiline", train_success_list],             
                },
                "agents test": {        
                    "test_ret": ["Multiline", test_ret_list],
                    "test_success": ["Multiline", test_success_list], 
                },
            }

            self.writer = SummaryWriter(log_dir = os.path.join(self.current_dir,self.logdir,self.logname,self.trainid,"runs"))
            self.writer.add_custom_scalars(layout)
    
    def handle_ancient_versions(self):
        if "buffer_num" not in self.config['buffer']: 
            self.config['buffer']['buffer_num'] = 1
        if 'tester2' not in self.config:
            self.config['tester2'] = {}
            self.config['tester2']['bool'] = True
            self.config['tester2']['env_name'] = 'rlbench'
            self.config['tester2']['num_test2_episodes'] = max(int(self.config['tester']['num_test_episodes']/10),1)        
        if 'params' not in self.config["environment"]: 
            self.config["environment"]['params'] = {}
            self.config["environment"]['params']['pick'] = {}
            self.config["environment"]['params']['pick']['atol'] = 0.01
            self.config["environment"]['params']['pick']['rtol'] = 0.0
            self.config["environment"]['params']['place']  = {}
            self.config["environment"]['params']['place']['mean'] = 0.0
            self.config["environment"]['params']['place']['std'] = 0.01
            self.config["environment"]['params']['reward']  = {}
            self.config["environment"]['params']['reward']['atol'] = 0.01
            self.config["environment"]['params']['reward']['rtol'] = 0.0

    def check_config_values(self):
        assert self.config['buffer']['buffer_num'] >= 1
        assert self.config['agent']['agent_num'] >= 1
        assert self.config['agent']['agent_num'] >= self.config['buffer']['buffer_num']

    def compute_and_replace_auto_values(self):
        self.task_name = self.config['environment']['task']['name']
        self.task_params = self.config['environment']['task']['params']
        self.action_space = self.config['agent']['action_space']
        self.state_space = self.config['environment']['state_space']

        ## OBS DIM
        if self.config['environment']['obs_dim'] == "auto":
            # RLBENCH JOINT
            if self.config['environment']['name'] == "rlbenchjoint":
                if self.task_name == "reach_target_no_distractors":
                    self.config['environment']['obs_dim'] = 3
            # GYM
            elif self.config['environment']['name'] == "gym":
                if self.task_name == "MountainCarContinuous-v0":
                    self.config['environment']['obs_dim'] = 2
                elif self.task_name == "InvertedPendulum-v4":
                    self.config['environment']['obs_dim'] = 4
                elif self.task_name == "InvertedDoublePendulum-v4":
                    self.config['environment']['obs_dim'] = 11
                elif self.task_name == 'Ant-v4':
                    self.config['environment']['obs_dim'] = 27
                
                
            # RLBENCH
            elif self.config['environment']['name'] == "rlbench":
                if self.task_name == "stack_blocks":
                    if self.state_space == "xyz":
                        self.config['environment']['obs_dim'] = 3 + self.task_params[0] * 3 + self.task_params[1] * 3
                    elif self.state_space == "xyz_quat":
                        self.config['environment']['obs_dim'] = 7 + self.task_params[0] * 7 + self.task_params[1] * 7
                    elif self.state_space == "xyz_z90":
                        self.config['environment']['obs_dim'] = 4 + self.task_params[0] * 4 + self.task_params[1] * 4
            else:
                self.print_logfile("Obs dim could not be computed","error")
                assert False
            self.print_logfile("Obs dim is computed: " + str(self.config['environment']['obs_dim']))

        ## ACT DIM
        if self.config['environment']['act_dim'] == "auto":
            # RLBENCH JOINT
            if self.config['environment']['name'] == "rlbenchjoint":
                if self.action_space == "joint":
                    self.config['environment']['act_dim'] = 6
                elif self.action_space == "jointgripper":
                    self.config['environment']['act_dim'] = 7
            # GYM
            elif self.config['environment']['name'] == "gym":
                if self.task_name == "MountainCarContinuous-v0":
                    self.config['environment']['act_dim'] = 1
                elif self.task_name == "InvertedPendulum-v4":
                    self.config['environment']['act_dim'] = 1
                elif self.task_name == 'InvertedDoublePendulum-v4':
                    self.config['environment']['act_dim'] = 1
                elif self.task_name == 'Ant-v4':
                    self.config['environment']['act_dim'] = 8
                    
            # RLBENCH
            elif self.config['environment']['name'] == "rlbench":
                if self.task_name == "stack_blocks":
                    if self.action_space == "pick_and_place_2d":
                        self.config['environment']['act_dim'] = 4
                    elif self.action_space == "pick_and_place_3d":
                        self.config['environment']['act_dim'] = 6
                    elif self.action_space == "pick_and_place_3d_quat":
                        self.config['environment']['act_dim'] = 14
                    elif self.action_space == "pick_and_place_3d_z90":
                        self.config['environment']['act_dim'] = 8
            else:
                self.print_logfile("Act dim could not be computed","error")
                assert False
            self.print_logfile("Act dim is computed: " + str(self.config['environment']['act_dim']))
        
        ## BOUNDARY MAX
        if self.config['agent']['boundary_min'] == "auto":
            # RLBENCH JOINT
            if self.config['environment']['name'] == "rlbenchjoint":
                if self.action_space == "joint":
                    self.config['agent']['boundary_min'] = [-1.0,-1.0,-1.0,-1.0,-1.0,-1.0]
                elif self.action_space == "jointgripper":
                    self.config['agent']['boundary_min'] = [-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,0.0]
            # GYM
            elif self.config['environment']['name'] == "gym":
                if self.task_name == "MountainCarContinuous-v0":
                    self.config['agent']['boundary_min'] = [-1.0]
                elif self.task_name == "InvertedPendulum-v4":
                    self.config['agent']['boundary_min'] = [-3.0]
                elif self.task_name == "InvertedDoublePendulum-v4":
                    self.config['agent']['boundary_min'] = [-1.0]
                elif self.task_name == 'Ant-v4':
                    self.config['agent']['boundary_min'] = [-1.0]
            # RLBENCH
            elif self.config['environment']['name'] == "rlbench":
                if self.task_name == "stack_blocks":
                    if self.action_space == "pick_and_place_2d":
                        self.config['agent']['boundary_min'] = [0.1,-0.3,0.1,-0.3]
                    elif self.action_space == "pick_and_place_3d":
                        self.config['agent']['boundary_min'] = [0.1,-0.3,0.76,0.1,-0.3,0.76]
                    elif self.action_space == "pick_and_place_3d_quat":
                        self.config['agent']['boundary_min'] = [0.1,-0.3,0.76,0,0,0,0,0.1,-0.3,0.76,0,0,0,0]
                    elif self.action_space == "pick_and_place_3d_z90":
                        self.config['agent']['boundary_min'] = [0.1,-0.3,0.76,0.0,0.1,-0.3,0.76,0.0]
        
        ## BOUNDARY MAX
        if self.config['agent']['boundary_max'] == "auto":
            # RLBENCH JOINT
            if self.config['environment']['name'] == "rlbenchjoint":
                if self.action_space == "joint":
                    self.config['agent']['boundary_max'] = [1.0,1.0,1.0,1.0,1.0,1.0]
                elif self.action_space == "jointgripper":
                    self.config['agent']['boundary_max'] = [1.0,1.0,1.0,1.0,1.0,1.0,1.0]
            # GYM
            elif self.config['environment']['name'] == "gym":
                if self.task_name == "MountainCarContinuous-v0":
                    self.config['agent']['boundary_max'] = [1.0]
                elif self.task_name == "InvertedPendulum-v4":
                    self.config['agent']['boundary_max'] = [3.0]
                elif self.task_name == "InvertedDoublePendulum-v4":
                    self.config['agent']['boundary_max'] = [1.0]
                elif self.task_name == 'Ant-v4':
                    self.config['agent']['boundary_max'] = [1.0]
                    
            # RLBENCH
            elif self.config['environment']['name'] == "rlbench":
                if self.task_name == "stack_blocks":
                    if self.action_space == "pick_and_place_2d":
                        self.config['agent']['boundary_max'] = [0.35,0.3,0.35,0.3]
                    elif self.action_space == "pick_and_place_2d_z90":
                        self.config['agent']['boundary_min'] = [0.35,0.3,1.0,0.35,0.3,1.0]
                    elif self.action_space == "pick_and_place_3d":
                        self.config['agent']['boundary_max'] = [0.35,0.3,0.86,0.35,0.3,0.86]
                    elif self.action_space == "pick_and_place_3d_quat":
                        self.config['agent']['boundary_max'] = [0.35,0.3,0.86,1,1,1,1,0.35,0.3,0.86,1,1,1,1]
                    elif self.action_space == "pick_and_place_3d_z90":
                        self.config['agent']['boundary_max'] = [0.35,0.3,0.86,1.0,0.35,0.3,0.86,1.0]

    def print_logfile(self,message,level = "info", terminal = True):
        if terminal:
            print("["+level+"]: " + str(message))
        if level == "debug":
            self.pylogger.debug(str(message))
        elif level == "warning":
            self.pylogger.warning(str(message))
        elif level == "error":
            self.pylogger.error(str(message))
        else:
            self.pylogger.info(str(message))

    def get_model_epoch(self,epoch,agent_id = 0):
        models = self.list_model_dir(agent_id)
        
        for model in models:
            end_ptr = model.find('_pi',6)
            if end_ptr > 6:
                model_num = int(model[6:end_ptr])    
                if model_num == epoch:
                    return model[:end_ptr]
        return None       

    def list_model_dir(self, agent_id = 0):
        return os.listdir(os.path.join(self.current_dir,self.logdir, self.logname,self.trainid,"model_backup",str(agent_id)))

    def get_model_path(self,name, agent_id = 0):
        return os.path.join(self.current_dir,self.logdir, self.logname,self.trainid,"model_backup",str(agent_id),name)
    
    def get_model_path_for_restart(self,dir = "model_backup_restart", epoch = 1, agent_id = 0):
        name = "model_" + str(epoch)
        return os.path.join(self.current_dir,self.logdir, self.logname,self.trainid,str(dir),str(agent_id),name)

    def get_config(self):
        return self.config
    
    def tb_writer_add_scalar(self,name,value,iter):
        self.writer.add_scalar(name, value, iter)
    
    def tb_writer_add_scalars(self,name,value,iter):
        self.writer.add_scalars(name, value, iter)
    
    def tb_writer_add_image(self,name,img, iter, dataformats='HWC'):
        self.writer.add_image(name, img, iter, dataformats=dataformats)

    def save_model(self,model,epoch,agent_id = 0):
        model_path = os.path.join(self.current_dir, self.logdir, self.logname,self.trainid,"model_backup",str(agent_id),"model_" + str(epoch))
        torch.save(model, model_path)
    
    def get_model_save_path(self,epoch,agent_id = 0):
        return os.path.join(self.current_dir, self.logdir, self.logname,self.trainid,"model_backup",str(agent_id),"model_" + str(epoch))

    
    # def save_replay_buffer(self, replay_buffer, epoch):
    #     path = os.path.join(self.current_dir, self.logdir, self.logname,self.trainid,"replay_buffer_backup","buffer_" + str(epoch)+".yaml")
    #     replay_buffer_copy = replay_buffer
    #     self.save_yaml(path, replay_buffer_copy)
    
    # def load_replay_buffer(self,epoch):
    #     path = os.path.join(self.current_dir, self.logdir, self.logname,self.trainid,"replay_buffer_backup","buffer_" + str(epoch)+".yaml")
    #     return self.load_yaml(path)
        
    def demo_exists(self,name):
        return os.path.isfile(os.path.join(self.current_dir, self.demodir, name + ".yaml"))

    def save_demos(self,name,demos):
        self.create_folder(os.path.join(self.current_dir, self.demodir))
        self.save_yaml(os.path.join(self.current_dir, self.demodir, name + ".yaml"), demos)
    
    def load_demos(self,name):
        return self.load_yaml(os.path.join(self.current_dir,self.demodir,name + ".yaml"))
    
    def remove_old_demo(self,name):
        os.remove(os.path.join(self.current_dir,self.demodir,name + ".yaml")) 
    
    # def compose_heatmap_color_image(self,heatmap_raw):
    #     #sample_num = np.sum(heatmap_raw)
    #     heatmap_rescaled = cv2.normalize(heatmap_raw, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    #     heatmap_img = cv2.applyColorMap(heatmap_rescaled, cv2.COLORMAP_JET)

        
    
    def tb_save_train_data_v2(self,
                              loss_q,
                              loss_pi,
                              train_ret,
                              train_ep_len,
                              train_ep_success,
                              env_error_num,
                              out_of_bounds_num,
                              reward_bonus_num,
                              demo_ratio,
                              heatmap_bool_pick,
                              heatmap_bool_place,
                              t,
                              actual_time,
                              update_iter):
        self.tb_writer_add_scalar("train/loss_q", loss_q, update_iter)
        self.tb_writer_add_scalar("train/loss_p", loss_pi, update_iter)
        self.tb_writer_add_scalar("train/train_ret", train_ret, t)
        self.tb_writer_add_scalar("train/train_ep_len", train_ep_len, t)
        self.tb_writer_add_scalar("train/train_ep_success", train_ep_success, t)
        env_error_num_ratio = (env_error_num / float(t))
        self.tb_writer_add_scalar("train/env_error_ratio", env_error_num_ratio, update_iter)
        out_of_bounds_ratio = (out_of_bounds_num / float(t))
        self.tb_writer_add_scalar("train/out_of_bounds_ratio", out_of_bounds_ratio, update_iter) 
        rel_time = 1000 * (actual_time / float(t))
        self.tb_writer_add_scalar("train/time_sec_1000_transitions", rel_time, t)
        reward_bonus_ratio = (reward_bonus_num / float(t))
        self.tb_writer_add_scalar("train/reward_bonus_ratio", reward_bonus_ratio, t)
        self.tb_writer_add_scalar("train/demo_ratio", demo_ratio, t)

        if self.heatmap_bool:
            # heatmap_bool_pick_ratio = heatmap_bool_pick / np.sum(heatmap_bool_pick)
            # heatmap_bool_place_ratio = heatmap_bool_place / np.sum(heatmap_bool_place)

            # heatmap_bool_pick_diff = heatmap_bool_pick - self.heatmap_bool_pick_old
            # heatmap_bool_place_diff = heatmap_bool_place - self.heatmap_bool_place_old

            # self.heatmap_bool_pick_old = np.copy(heatmap_bool_pick)
            # self.heatmap_bool_place_old = np.copy(heatmap_bool_place)

            heatmap_bool_pick_norm = heatmap_bool_pick / np.max(heatmap_bool_pick)
            heatmap_bool_place_norm = heatmap_bool_place / np.max(heatmap_bool_place)

            # heatmap_bool_pick_diff_norm = heatmap_bool_pick_diff / np.max(heatmap_bool_pick_diff)
            # heatmap_bool_place_diff_norm = heatmap_bool_place_diff / np.max(heatmap_bool_place_diff)

            self.tb_writer_add_image("sampler/hetmap_pick_all",heatmap_bool_pick_norm, t, dataformats='HW')
            self.tb_writer_add_image("sampler/hetmap_place_all",heatmap_bool_place_norm, t, dataformats='HW')

            # self.tb_writer_add_image("sampler/hetmap_pick_last",heatmap_bool_pick_diff_norm, t, dataformats='HW')
            # self.tb_writer_add_image("sampler/hetmap_place_last",heatmap_bool_place_diff_norm, t, dataformats='HW')
    
    def np2dict(self,data_np):
        data = {}
        for agent_id in range(data_np.shape[0]):
            data["agent_" + str(agent_id)]= data_np[agent_id]        
        return data

    def tb_save_train_data_v3(self,
                              loss_q_np,
                              loss_pi_np,
                              train_ret_np,
                              train_ep_len_np,
                              train_ep_success_np,
                              env_error_num,
                              out_of_bounds_num,
                              reward_bonus_num,
                              demo_ratio,
                              heatmap_pick,
                              heatmap_place,
                              t,
                              actual_time,
                              update_iter):
        
        for i in range(self.agent_num):
            self.writer.add_scalar("train/train_ret_"+ str(i), train_ret_np[i], t)
            self.writer.add_scalar("train/train_ep_len_"+ str(i), train_ep_len_np[i], t)
            self.writer.add_scalar('train/loss_q_'+ str(i), loss_q_np[i], update_iter)
            self.writer.add_scalar("train/loss_pi_"+ str(i), loss_pi_np[i], update_iter)
            self.writer.add_scalar("train/train_success_"+ str(i), train_ep_success_np[i], t)

        env_error_num_ratio = (env_error_num / float(t))
        self.tb_writer_add_scalar("train_glob/env_error_ratio", env_error_num_ratio, update_iter)
        out_of_bounds_ratio = (out_of_bounds_num / float(t))
        self.tb_writer_add_scalar("train_glob/out_of_bounds_ratio", out_of_bounds_ratio, update_iter) 
        rel_time = 1000 * (actual_time / float(t))
        self.tb_writer_add_scalar("train_glob/time_sec_1000_transitions", rel_time, t)
        reward_bonus_ratio = (reward_bonus_num / float(t))
        self.tb_writer_add_scalar("train_glob/reward_bonus_ratio", reward_bonus_ratio, t)
        self.tb_writer_add_scalar("train_glob/demo_ratio", demo_ratio, t)

        if self.heatmap_bool:

            heatmap_pick_norm = heatmap_pick / np.max(heatmap_pick)
            heatmap_place_norm = heatmap_place / np.max(heatmap_place)

            self.tb_writer_add_image("sampler/hetmap_pick_all",heatmap_pick_norm, t, dataformats='HW')
            self.tb_writer_add_image("sampler/hetmap_place_all",heatmap_place_norm, t, dataformats='HW')

     
    
    def tb_save_train_data(self,loss_q,loss_pi,sum_ep_len,sum_ep_ret,episode_iter,env_error_num,t,log_loss_iter):
        self.tb_writer_add_scalar("train/loss_q", loss_q, log_loss_iter)
        self.tb_writer_add_scalar("train/loss_p", loss_pi, log_loss_iter)
        if episode_iter > 0 :
            avg_sum_ep_len = sum_ep_len / float(episode_iter)
            avg_sum_ep_ret = sum_ep_ret / float(episode_iter)
        else:
            avg_sum_ep_len = 0
            avg_sum_ep_ret = 0
        env_error_num_percentage = (env_error_num / t) * 100
        self.tb_writer_add_scalar("train/avg_return_since_last_epoch", avg_sum_ep_ret, log_loss_iter)
        self.tb_writer_add_scalar("train/avg_episode_length_since_last_epoch", avg_sum_ep_len, log_loss_iter)
        self.tb_writer_add_scalar("train/env_error_%", env_error_num_percentage, log_loss_iter) 

    def save_eval_range(self,data,epoch):
        name = self.logname + "_plot_range_" + str(epoch)
        dir = os.path.join(self.current_dir, self.logdir, self.logname,self.trainid,"plots_raw_data")

        with open(os.path.join(dir,name + ".npy" ),'wb') as f:
            np.save(f, data)    
        
        # plot = True
        # if plot == True:
        #     import matplotlib.pyplot as plt

        #     inputs_np = data[:,0]
        #     outputs_np = data[:,1]

        #     plt.plot(inputs_np,inputs_np, color = "blue", label="ground-truth")
        #     plt.plot(inputs_np,outputs_np, color = "orange",label="prediction")
        #     #plt.title("Results in range " +  args.trainname)
        #     plt.title("Results in range - " + str(epoch))
        #     plt.legend()
        #     plt.savefig(os.path.join(dir,name + ".png" ))
        #     #plt.show()
        #     plt.clf()

    # File management ##############################################################
    def create_folder(self,path):
        if not os.path.exists(path):
            os.makedirs(path)
            print(path + ' folder is created!')
        else:
            print(path + ' folder already exists!')

    def save_yaml(self, path, data):
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)

    def load_yaml(self, file):
        if file is not None:
            with open(file) as f:
                return yaml.load(f, Loader=yaml.UnsafeLoader)
        return None

    def save_test2(self,df,test2env):

        self.create_folder(os.path.join(self.current_dir, "csv_to_plot"))

        exp_name = str(self.config['general']['exp_name']) + "_" + str(self.trainid)
        file_name = os.path.join(self.current_dir,"csv_to_plot","test_epochs_"+exp_name+"_"+test2env+".csv")

        #df.to_excel(file_name) 
        df.to_csv(file_name,index=False)

    # Save Images, Ground-Truth #####################################################

    def save_camera_images(self, observation, front = True, wrist = True, base_name = "temp", iter = 0):

        name_core = base_name + "_" + str(iter)
        if front:
            # FRONT CAMERA RGBobservation
            cam_front_rgb = Image.fromarray(observation.front_rgb)
            cam_front_rgb.save(self.current_dir + "/" + self.logdir + "/" + name_core + "_front_rgb.png")

            # FRONT CAMERA DEPTH
            cam_front_depth = Image.fromarray(observation.front_depth)   
            cam_front_depth_np = np.array(cam_front_depth)
            with open(self.current_dir + "/" + self.logdir + "/" + name_core +"_front_depth.npy", 'wb') as f:
                np.save(f, cam_front_depth_np)
            
            # FRONT CAMERA POINT CLOUD
            with open(self.current_dir + "/" + self.logdir + "/" + name_core +"_front_point_cloud.npy", 'wb') as f:
                np.save(f, observation.front_point_cloud)
        
        if wrist:
            # FRONT CAMERA RGBobservation
            cam_wrist_rgb = Image.fromarray(observation.wrist_rgb)
            cam_wrist_rgb.save(self.current_dir + "/" + self.logdir + "/" + name_core + "_wrist_rgb.png")

            # FRONT CAMERA DEPTH
            cam_wrist_depth = Image.fromarray(observation.wrist_depth)   
            cam_wrist_depth_np = np.array(cam_wrist_depth)
            with open(self.current_dir + "/" + self.logdir + "/" + name_core +"_wrist_depth.npy", 'wb') as f:
                np.save(f, cam_wrist_depth_np)
            
            # FRONT CAMERA POINT CLOUD
            with open(self.current_dir + "/" + self.logdir + "/" + name_core +"_wrist_point_cloud.npy", 'wb') as f:
                np.save(f, observation.wrist_point_cloud)

    def save_objects_positions(self, task_env, observation,base_name = "temp", iter = 0):
        name_core = base_name + "_" + str(iter)

        objs = task_env._scene.task._graspable_objects 
        
        objs_poses_xyz = np.empty((0,3), np.float64)
        objs_poses_pixel = np.empty((0,2), np.uint32)
        for obj in objs:
            obj_xyz = obj.get_position()
            obj_xyz = np.reshape(obj_xyz, (-1, 3))
            objs_poses_xyz = np.append(objs_poses_xyz, obj_xyz, axis=0)

            # Pixel - need to simplify
            xyz_homo = np.hstack((obj_xyz, np.ones((obj_xyz.shape[0],1))))
            xcam = [np.linalg.inv(observation.misc['front_camera_extrinsics']).dot(x) for x in xyz_homo]
            xpix = [observation.misc['front_camera_intrinsics'].dot(x[:-1]) for x in xcam]
            # print(xpix)
            # xpix0 = xpix[0]
            # print(xpix0)
            # pix_coords = np.array([xpix0[0]/xpix0[2],xpix0[1]/xpix0[2]], np.uint32)
            x_list = [int(x[0]/x[2]) for x in xpix]
            y_list = [int(x[1]/x[2]) for x in xpix]
            pix_coords = np.array([x_list[0],y_list[0]],np.uint32)
            pix_coords = np.reshape(pix_coords, (-1, 2))
            objs_poses_pixel = np.append(objs_poses_pixel, pix_coords, axis=0)
        
        # print(objs_poses_xyz)
        # print(objs_poses_pixel)

        with open(self.current_dir + "/" + self.logdir + "/" + name_core +"_objects_pos_xyz.npy", 'wb') as f:
            np.save(f, objs_poses_xyz)
        
        with open(self.current_dir + "/" + self.logdir + "/" + name_core +"_objects_pos_pixel.npy", 'wb') as f:
            np.save(f, objs_poses_pixel)