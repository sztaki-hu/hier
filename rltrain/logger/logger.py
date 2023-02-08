import os
import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.tensorboard import SummaryWriter


class Logger:
    # init method or constructor
    def __init__(self, current_dir,config_path, trainid, light_mode = False):
        self.current_dir = current_dir
        self.config_path = config_path
        self.trainid = str(trainid)

        self.config = self.load_yaml(self.config_path)

        self.logdir = self.config['general']['logdir']
        self.logname = self.config['general']['exp_name'] + "_" + self.config['environment']['task']['name'] + "_" + self.config['agent']['type'] 

        self.demodir = self.config['general']['demodir']

        cfg_rlbench = {'path' : config_path}
        self.create_folder(os.path.join(self.current_dir, "cfg_rlbench"))

        self.compute_and_replace_auto_values()

        self.save_yaml(os.path.join(self.current_dir, "cfg_rlbench" ,"config.yaml"),cfg_rlbench)

        # cfg_rlbench_2 = self.load_yaml(os.path.join(self.current_dir, "cfg_rlbench" ,"config.yaml"))
        # print(cfg_rlbench_2)
        # print(cfg_rlbench_2['path'])

        if light_mode == False:
            self.create_folder(os.path.join(self.current_dir,self.logdir, self.logname,self.trainid,"model_backup"))
            #self.create_folder(os.path.join(self.current_dir, self.logdir, self.logname,self.trainid,"plots_raw_data"))
            self.save_yaml(os.path.join(self.current_dir, self.logdir,self.logname,self.trainid,"config.yaml"),self.config)
            self.writer = SummaryWriter(log_dir = os.path.join(self.current_dir,self.logdir,self.logname,self.trainid,"runs"))

            
    def compute_and_replace_auto_values(self):
        self.task_name = self.config['environment']['task']['name']
        self.task_params = self.config['environment']['task']['params']
        self.action_space = self.config['agent']['action_space']

        if self.config['environment']['obs_dim'] == "auto":
            if self.task_name == "stack_blocks":
                self.config['environment']['obs_dim'] = 3 + self.task_params[0] * 3 + self.task_params[1] * 3
            else:
                print("[ERROR] Obs dim could not be computed")
                assert False
            print("Obs dim is computed: " + str(self.config['environment']['obs_dim']))
        if self.config['environment']['act_dim'] == "auto":
            if self.action_space == "pick_and_place_3d":
                self.config['environment']['act_dim'] = 6
            else:
                print("[ERROR] Act dim could not be computed")
                assert False
            print("Act dim is computed: " + str(self.config['environment']['act_dim']))



    def new_model_to_test(self,epoch):
        models = self.list_model_dir()
        for model in models:
            model_num = int(model[6:])    
            if model_num == epoch:
                return model
        return None

    def list_model_dir(self):
        return os.listdir(os.path.join(self.current_dir,self.logdir, self.logname,self.trainid,"model_backup"))

    def get_model_path(self,name):
        return os.path.join(self.current_dir,self.logdir, self.logname,self.trainid,"model_backup",name)

    def get_config(self):
        return self.config
    
    def tb_writer_add_scalar(self,name,value,iter):
        self.writer.add_scalar(name, value, iter)

    def save_model(self,model,epoch):
        model_path = os.path.join(self.current_dir, self.logdir, self.logname,self.trainid,"model_backup","model_" + str(epoch))
        torch.save(model, model_path)
    
    def get_model_save_path(self,epoch):
        return os.path.join(self.current_dir, self.logdir, self.logname,self.trainid,"model_backup","model_" + str(epoch))
        
    def demo_exists(self,name):
        return os.path.isfile(os.path.join(self.current_dir, self.demodir, name + ".yaml"))

    def save_demos(self,name,demos):
        self.create_folder(os.path.join(self.current_dir, self.demodir))
        self.save_yaml(os.path.join(self.current_dir, self.demodir, name + ".yaml"), demos)
    
    def load_demos(self,name):
        return self.load_yaml(os.path.join(self.current_dir,self.demodir,name + ".yaml"))
    
    def remove_old_demo(self,name):
        os.remove(os.path.join(self.current_dir,self.demodir,name + ".yaml")) 
    
    def tb_save_train_data_v2(self,loss_q,loss_pi,env_error_num,out_of_bounds_num,t,actual_time,update_iter):
        self.tb_writer_add_scalar("train/loss_q", loss_q, update_iter)
        self.tb_writer_add_scalar("train/loss_p", loss_pi, update_iter)
        env_error_num_ratio = (env_error_num / float(t))
        self.tb_writer_add_scalar("train/env_error_ratio", env_error_num_ratio, update_iter)
        out_of_bounds_ratio = (out_of_bounds_num / float(t))
        self.tb_writer_add_scalar("train/out_of_bounds_ratio", out_of_bounds_ratio, update_iter) 
        rel_time = 1000 * (actual_time / float(t))
        self.tb_writer_add_scalar("train/time_sec_1000_transitions", rel_time, t)
    
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