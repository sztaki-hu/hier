import os
import numpy as np
import yaml
import time
import random
import logging
import pandas as pd
from collections import defaultdict
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from PIL import Image
from torch.utils.tensorboard import SummaryWriter


class Logger:
    # init method or constructor
    def __init__(self, current_dir, main_args, display_mode = False, exp = None):

        if display_mode: self.seed_id = str(main_args.seedid)

        # Load config and set up main variables
        self.current_dir = current_dir
        self.config_path = os.path.join(current_dir,main_args.config) if display_mode == False else os.path.join(current_dir,main_args.config,self.seed_id,"config.yaml")             
        self.config = self.load_yaml(self.config_path)
        self.config_framework = self.load_yaml(os.path.join(current_dir,'cfg_framework','config_framework.yaml'))
        self.config_tasks = self.load_yaml(os.path.join(current_dir,'cfg_framework','config_tasks.yaml'))
        self.config['general']['current_dir'] = current_dir

        # Replace 'input' values of config file if it is main_auto.py 
        if exp != None:
            self.config['environment']['name'] = exp['env']
            self.config['environment']['task']['name'] = exp['task']
            self.config['agent']['type'] = exp['agent']
            self.config['buffer']['her']['goal_selection_strategy'] = exp['her_strategy']

            if exp['cl'] == 'nocl': self.config['trainer']['mode'] = 'normal'
            if exp['cl'] in ['linear','sqrt','quad']:     
                self.config['trainer']['mode'] = 'cl'     
                self.config['trainer']['cl']['type'] = 'predefined'         
                self.config['trainer']['cl']['predefined']['pacing_profile'] = exp['cl']
            if exp['cl'] == 'selfpaced': 
                self.config['trainer']['mode'] = 'cl'   
                self.config['trainer']['cl']['type'] = 'selfpaced'
            if exp['cl'] == 'selfpaceddual': 
                self.config['trainer']['mode'] = 'cl'   
                self.config['trainer']['cl']['type'] = 'selfpaceddual'
            if exp['cl'] == 'controldiscrete': 
                self.config['trainer']['mode'] = 'cl'   
                self.config['trainer']['cl']['type'] = 'controldiscrete'

            exp_name = self.config['general']['exp_name']
            for key in list(exp.keys()):
                if key != 'env':
                    exp_name += "_"
                    exp_name += exp[key]         
            self.config['general']['exp_name'] = exp_name
        
        # Lognames
        self.logdir = self.config['general']['logdir']
        self.exp_name = self.config['general']['exp_name']
        print(self.config['general']['exp_name'])
        
        # Compute and replace auto values
        self.compute_and_replace_auto_values()
        
        # Demos
        self.demodir = self.config['general']['demodir']

        # Create log folders and files
        self.exp_folder = os.path.join(self.current_dir,self.logdir, self.exp_name)

        if display_mode == False: 
            
            # Exp folder and get seed id    
            self.create_folder(os.path.join(self.exp_folder))
            self.seed_id = str(len(os.listdir(self.exp_folder)))
            self.create_folder(os.path.join(self.exp_folder,self.seed_id))

            # Save config
            self.save_yaml(os.path.join(self.exp_folder,self.seed_id,"config.yaml"),self.config)
            
            # Backup model folder
            self.create_folder(os.path.join(self.exp_folder,self.seed_id,"model_backup"))    

            # Tensorboard
            self.tb_logdir = os.path.join(self.exp_folder,self.seed_id,"runs")
            self.writer = SummaryWriter(log_dir = self.tb_logdir)     
        
        # Printout logging
        log_file_path = os.path.join(self.exp_folder,self.seed_id,'logs.log')
        if os.path.isfile(log_file_path):
            os.remove(log_file_path) 

        logging.basicConfig(filename=log_file_path,level=logging.DEBUG)
        self.pylogger = logging.getLogger('mylogger')

        # Set up RLBench path
        # cfg_rlbench = {'path' : self.config_path}
        # self.create_folder(os.path.join(self.current_dir, "cfg_rlbench"))
        # self.save_yaml(os.path.join(self.current_dir, "cfg_rlbench" ,"config.yaml"),cfg_rlbench)

    def compute_and_replace_auto_values(self):
        env_name = self.config['environment']['name']
        task_name = self.config['environment']['task']['name']

        ## MAX_EP_LEN
        if self.config['sampler']['max_ep_len'] == "auto":  
            self.config['sampler']['max_ep_len'] = self.config_tasks[env_name][task_name]['max_ep_len']               

        ## OBS DIM
        if self.config['environment']['obs_dim'] == "auto":
            self.config['environment']['obs_dim']  = self.config_tasks[env_name][task_name]['obs_dim']

        ## ACT DIM
        if self.config['environment']['act_dim'] == "auto":
            self.config['environment']['act_dim'] = self.config_tasks[env_name][task_name]['act_dim']
        
        ## BOUNDARY MIN
        if self.config['agent']['boundary_min'] == "auto":
            self.config['agent']['boundary_min'] = self.config_tasks[env_name][task_name]['boundary_min']
        
        ## BOUNDARY MAX
        if self.config['agent']['boundary_max'] == "auto":
            self.config['agent']['boundary_max'] = self.config_tasks[env_name][task_name]['boundary_max']
        
        if self.config['general']['seed'] == 'random': self.config['general']['seed'] = random.randint(0,1000)


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
    
    # Config

    def get_config(self):
        return self.config
    
    def get_config_framework(self):
        return self.config_framework
    
    # TB
    
    def tb_writer_add_scalar(self,name,value,iter):
        self.writer.add_scalar(name, value, iter)
    
    def tb_writer_add_scalars(self,name,value,iter):
        self.writer.add_scalars(name, value, iter)
    
    def tb_writer_add_image(self,name,img, iter, dataformats='HWC'):
        self.writer.add_image(name, img, iter, dataformats=dataformats)
    
    # Models
    
    def get_model_save_path(self,epoch,best_model=False):
        if best_model == False:
            return os.path.join(self.exp_folder,self.seed_id,"model_backup","model_" + str(epoch))
        else:
            return os.path.join(self.exp_folder,self.seed_id,"model_backup","model_best_model")
    
    # Demos

    def demo_exists(self,name):
        return os.path.isfile(os.path.join(self.current_dir, self.demodir, name + ".yaml"))

    def save_demos(self,name,demos):
        self.create_folder(os.path.join(self.current_dir, self.demodir))
        self.save_yaml(os.path.join(self.current_dir, self.demodir, name + ".yaml"), demos)
    
    def load_demos(self,name):
        return self.load_yaml(os.path.join(self.current_dir,self.demodir,name + ".yaml"))
    
    def remove_old_demo(self,name):
        os.remove(os.path.join(self.current_dir,self.demodir,name + ".yaml")) 

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
    
    
    def tb_tabulate_events(self,dpath):
        summary_iterators = [EventAccumulator(os.path.join(dpath, dname)).Reload() for dname in os.listdir(dpath)]

        # print(summary_iterators[0].Tags())

        # assert False

        tags = summary_iterators[0].Tags()['scalars']

        for it in summary_iterators:
            assert it.Tags()['scalars'] == tags

        out = defaultdict(list)
        steps = []

        for tag in tags:
            steps = [e.step for e in summary_iterators[0].Scalars(tag)]

            for events in zip(*[acc.Scalars(tag) for acc in summary_iterators]):
                assert len(set(e.step for e in events)) == 1

                out[tag].append([e.value for e in events])

        return out, steps

    def tb_to_csv(self,dpath):
        dirs = os.listdir(dpath)

        d, steps = self.tb_tabulate_events(dpath)
        tags, values = zip(*d.items())
        np_values = np.array(values)

        for index, tag in enumerate(tags):
            df = pd.DataFrame(np_values[index], index=steps, columns=dirs)
            df.to_csv(self.tb_get_file_path(dpath, tag))


    def tb_get_file_path(self,dpath, tag):
        file_name = tag.replace("/", "_") + '.csv'
        folder_path = os.path.join(dpath, 'csv')
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        print(folder_path)
        return os.path.join(folder_path, file_name)
    
    
    def tb_close(self):
        self.writer.flush()
        self.writer.close()
        self.tb_to_csv(self.tb_logdir)