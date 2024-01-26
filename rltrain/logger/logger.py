import os
import numpy as np
import yaml
import time
import random
import logging
import pandas as pd
from collections import defaultdict
from typing import Dict, Union, Optional, Tuple
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from PIL import Image
# from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.writer import SummaryWriter

class Logger:
    # init method or constructor
    def __init__(self, 
                 current_dir: str, 
                 configpath: str,
                 display_mode: bool = False, 
                 exp: Optional[Dict] = None, 
                 is_test_config: bool = False,
                 seed: int = 0
                 ) -> None:

        if display_mode: self.seed_id = str(seed)
        elif is_test_config: self.seed_id = "0"

        # Load config and set up main variables
        self.current_dir = current_dir
        self.config_path = os.path.join(current_dir,configpath) if display_mode == False else os.path.join(current_dir,configpath,self.seed_id,"config.yaml")             
        self.config = self.load_yaml(self.config_path)
        self.config_framework = self.load_yaml(os.path.join(current_dir,'cfg_framework','config_framework.yaml'))
        self.config_tasks = self.load_yaml(os.path.join(current_dir,'cfg_framework','config_tasks.yaml'))
        self.config['general']['current_dir'] = current_dir

        # Replace 'input' values of config file if it is main_auto.py 
        if exp != None:
            # Task
            self.config['environment']['name'] = exp['main']['env']
            self.config['environment']['task']['name'] = exp['main']['task']
            # Agent
            self.config['agent']['type'] = exp['main']['agent']
            self.config['agent']['sac']['alpha'] = exp['main']['agent_sac_alpha']
            self.config['agent']['gamma'] = exp['main']['agent_gamma']
            self.config['agent']['learning_rate'] = exp['main']['agent_learning_rate']
            # Env
            self.config['environment']['reward']['reward_shaping_type'] = exp['main']['reward_shaping_type']
            self.config['environment']['reward']['reward_bonus'] = exp['main']['reward_bonus']
            # Buffer
            self.config['buffer']['replay_buffer_size'] = exp['main']['replay_buffer_size']
            self.config['buffer']['her']['goal_selection_strategy'] = exp['main']['her_strategy']
            self.config['buffer']['hier']['buffer_size'] = exp['main']['hier_buffer_size']
            self.config['buffer']['hier']['lambda']['mode'] = exp['main']['hier_lambda_mode']
            self.config['buffer']['hier']['lambda']['fix']['lambda'] = exp['main']['hier_lambda_fix_lambda']
            self.config['buffer']['hier']['lambda']['predefined']['lambda_start'] = exp['main']['hier_lambda_predefined_lambda_start']
            self.config['buffer']['hier']['lambda']['predefined']['lambda_end'] = exp['main']['hier_lambda_predefined_lambda_end']
            self.config['buffer']['hier']['xi']['mode'] = exp['main']['hier_xi_mode']
            self.config['buffer']['hier']['xi']['xi'] = exp['main']['hier_xi_xi'] 
            self.config['buffer']['per']['mode'] = exp['main']['per_mode']
            # Trainer
            self.config['trainer']['total_timesteps'] = exp['main']['trainer_total_timesteps']
            # Eval
            self.config['eval']['freq'] = exp['main']['eval_freq']
            self.config['eval']['num_episodes'] = exp['main']['eval_num_episodes']

            if self.config['buffer']['per']['mode'] != 'noper' and self.config['buffer']['hier']['xi']['set_prioritized_for_PER']: 
                self.config['buffer']['hier']['xi']['mode'] = 'prioritized'
                exp['main']['hier_xi_mode'] = 'prioritized'
            
            # ISE
            self.config['trainer']['init_state']['ise']['type'] = exp['main']['ise']
            self.config['trainer']['init_state']['ise']['range_growth_mode'] = exp['main']['ise_range_growth_mode']    

            # Sort dict for logging
            myKeys = list(exp['main'].keys())
            myKeys.sort()
            exp['main'] = {i: exp['main'][i] for i in myKeys}

            # Create name
            exp_name = self.config['general']['exp_name']
            for key in list(exp['main'].keys()):
                print(key)
                if exp['exp_in_name'][key]:
                    exp_name += "_"
                    if type(exp['main'][key]) == float:
                        str_num = str(exp['main'][key])
                        str_num = str_num.replace('.','')
                        str_num = exp['exp_abb'][key] + str_num
                        exp_name += str_num
                    else:
                        exp_name += exp['main'][key]         
            self.config['general']['exp_name'] = exp_name
        
        # Lognames
        self.logdir = self.config['general']['logdir']
        self.exp_name = self.config['general']['exp_name']
        
        # Compute and replace auto values
        self.compute_and_replace_auto_values()
        if display_mode == False:
            if self.config['buffer']['per']['mode'] != 'noper' and self.config['buffer']['hier']['xi']['set_prioritized_for_PER']: 
                self.config['buffer']['hier']['xi']['mode'] = 'prioritized'
        
        # Demos
        self.demodir = self.config['general']['demodir']

        # Create log folders and files
        self.exp_folder = os.path.join(self.current_dir,self.logdir, self.exp_name)

        if display_mode == False and is_test_config == False: 
            
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
            self.log_file_path = os.path.join(self.exp_folder,self.seed_id,'logs.log')
            if os.path.isfile(self.log_file_path):
                os.remove(self.log_file_path) 
            
            self.print_logfile(message = "Logger is ready", level = "info", terminal = False) 
            self.print_logfile(message = self.exp_name, level = "info", terminal = False) 
        


        # logging.basicConfig(filename=log_file_path,level=logging.DEBUG)
        # self.pylogger = logging.getLogger('mylogger')

        

        # Set up RLBench path
        # cfg_rlbench = {'path' : self.config_path}
        # self.create_folder(os.path.join(self.current_dir, "cfg_rlbench"))
        # self.save_yaml(os.path.join(self.current_dir, "cfg_rlbench" ,"config.yaml"),cfg_rlbench)

    def compute_and_replace_auto_values(self) -> None:
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


    # def print_logfile(self, message: str, level: str = "info", terminal: bool = True) -> None:
    #     if terminal:
    #         print("["+level+"]: " + str(message))
    #     if level == "debug":
    #         self.pylogger.debug(str(message))
    #     elif level == "warning":
    #         self.pylogger.warning(str(message))
    #     elif level == "error":
    #         self.pylogger.error(str(message))
    #     else:
    #         self.pylogger.info(str(message))
    
    def print_logfile(self, 
                      message: str, 
                      level: str = "info", 
                      terminal: bool = True, 
                      display_mode: bool = False
                      ) -> None:
        if terminal or display_mode:
            print("["+level+"]: " + str(message))
        if display_mode == False:
            with open(self.log_file_path, 'a') as file1:
                file1.write(message + "\n")

    # Config ########################################################################

    def get_config(self) -> Dict:
        return self.config
    
    def get_config_framework(self) -> Dict:
        return self.config_framework
    
    # Models ########################################################################
    
    def get_model_save_path(self, model_name: Union[int,str]) -> str:
        return os.path.join(self.exp_folder,self.seed_id,"model_backup","model_" + str(model_name))

    # File management ##############################################################
    def create_folder(self, path: str) -> None:
        if not os.path.exists(path):
            os.makedirs(path)
            print(path + ' folder is created!')
        else:
            print(path + ' folder already exists!')

    def save_yaml(self, path: str, data: Dict) -> None:
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)

    def load_yaml(self, file: str) -> Dict:
        if file is not None:
            with open(file) as f:
                return yaml.load(f, Loader=yaml.UnsafeLoader)
        return {}

  
    # TB ############################################################################x
    
    def tb_writer_add_scalar(self, name: str, value: float, iter: int) -> None:
        self.writer.add_scalar(name, value, iter)
    
    # def tb_writer_add_scalars(self,name,value,iter):
    #     self.writer.add_scalars(name, value, iter)
    
    # def tb_writer_add_image(self,name,img, iter, dataformats='HWC'):
    #     self.writer.add_image(name, img, iter, dataformats=dataformats)
    
    def tb_tabulate_events(self, dpath: str) -> Tuple:
        summary_iterators = [EventAccumulator(os.path.join(dpath, dname)).Reload() for dname in os.listdir(dpath)]

        # print(summary_iterators[0].Tags())

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

    def tb_to_csv(self, dpath: str) -> None:
        dirs = os.listdir(dpath)

        d, steps = self.tb_tabulate_events(dpath)
        tags, values = zip(*d.items())
        np_values = np.array(values)

        for index, tag in enumerate(tags):
            df = pd.DataFrame(np_values[index], index=steps, columns=dirs)
            df.to_csv(self.tb_get_file_path(dpath, tag))


    def tb_get_file_path(self, dpath: str, tag) -> str:
        file_name = tag.replace("/", "_") + '.csv'
        folder_path = os.path.join(dpath, 'csv')
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        print(folder_path)
        return os.path.join(folder_path, file_name)
    
    
    def tb_close(self) -> None:
        self.writer.flush()
        self.writer.close()
        self.tb_to_csv(self.tb_logdir)