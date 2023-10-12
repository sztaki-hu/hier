from copy import deepcopy
import numpy as np
import torch
from torch.optim import Adam
import gym
import time

import rltrain.agents.ddpg.core as core

class Agent:
    def __init__(self,device,config):
        self.device = device

        self.actor_critic=core.MLPActorCritic
        self.seed = config['general']['seed'] 
        self.gamma = config['agent']['gamma'] 
        self.polyak = config['agent']['polyak']
        self.pi_lr = config['agent']['learning_rate']  # pi_lr and q_lr are the same bc of SB3 implementation
        self.q_lr = config['agent']['learning_rate']  # pi_lr and q_lr are the same bc of SB3 implementation

        self.obs_dim = config['environment']['obs_dim'] 
        self.act_dim = config['environment']['act_dim']
        self.boundary_min = np.array(config['agent']['boundary_min'])[:self.act_dim]
        self.boundary_max = np.array(config['agent']['boundary_max'])[:self.act_dim]

        # torch.manual_seed(self.seed)
        # np.random.seed(self.seed)

        self.act_offset = (self.boundary_min + self.boundary_max) / 2.0
        self.act_limit = (self.boundary_max - self.boundary_min) / 2.0

        self.act_offset = torch.from_numpy(self.act_offset).float().to(self.device)
        self.act_limit = torch.from_numpy(self.act_limit).float().to(self.device)

        # Create actor-critic module and target networks
        self.ac = self.actor_critic(self.obs_dim, self.act_dim, self.act_offset, self.act_limit, hidden_sizes=tuple(config['agent']['hidden_sizes']))

        self.ac.pi.to(self.device)
        self.ac.q.to(self.device)

        print(self.ac.pi)
        print(self.ac.q)

        self.ac_targ = deepcopy(self.ac)

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.ac_targ.parameters():
            p.requires_grad = False

        # Count variables (protip: try to get a feel for how different size networks behave!)
        self.var_counts = tuple(core.count_vars(module) for module in [self.ac.pi, self.ac.q])


        # Set up optimizers for policy and q-function
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=self.pi_lr)
        self.q_optimizer = Adam(self.ac.q.parameters(), lr=self.q_lr)

        if config['general']['init_weights']['bool']:
            self.init_weights_path = config['general']['init_weights']['path']
            self.init_weights_mode = config['general']['init_weights']['mode']
            self.load_weights(self.init_weights_path, mode=self.init_weights_mode, eval=False)


    # Set up function for computing DDPG Q-loss
    def compute_loss_q(self,data):

        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        o = o.float().to(self.device)
        a = a.to(self.device)
        r = r.to(self.device)
        o2 = o2.float().to(self.device)
        d = d.to(self.device)

        q = self.ac.q(o,a)

        weights_IS = data['weights']
        weights_IS = weights_IS.to(self.device)

        # Bellman backup for Q function
        with torch.no_grad():
            q_pi_targ = self.ac_targ.q(o2, self.ac_targ.pi(o2))
            backup = r + self.gamma * (1 - d) * q_pi_targ

        # MSE loss against Bellman backup
        loss_q = (((q - backup)**2)*weights_IS).mean()

        # Useful info for logging
        loss_info = dict(QVals=q.cpu().detach().numpy())

        # Priorities for PER
        td_error1 = q - backup
        batch_priorities = abs((td_error1 + 1e-5).squeeze()).detach().cpu().numpy()

        return loss_q, batch_priorities, loss_info

    # Set up function for computing DDPG pi loss
    def compute_loss_pi(self,data):
        o = data['obs'].to(self.device)
        q_pi = self.ac.q(o, self.ac.pi(o))
        return -q_pi.mean()


    def update(self,data,placeholder = None):
        # First run one gradient descent step for Q.
        self.q_optimizer.zero_grad()
        loss_q, batch_priorities, loss_info = self.compute_loss_q(data)
        loss_q.backward()
        self.q_optimizer.step()

        ret_loss_q = loss_q.item()

        # Freeze Q-network so you don't waste computational effort 
        # computing gradients for it during the policy learning step.
        for p in self.ac.q.parameters():
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        self.pi_optimizer.zero_grad()
        loss_pi = self.compute_loss_pi(data)
        loss_pi.backward()
        self.pi_optimizer.step()

        ret_loss_pi = loss_pi.item()

        # Unfreeze Q-network so you can optimize it at next DDPG step.
        for p in self.ac.q.parameters():
            p.requires_grad = True

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)
        
        return ret_loss_q, ret_loss_pi, batch_priorities

    def get_action(self,o, noise_scale):
        o = torch.from_numpy(o).float().unsqueeze(0).to(self.device)
        a = self.ac.act(o)[0]
        a += noise_scale * np.random.randn(self.act_dim)
        return np.clip(a, self.boundary_min, self.boundary_max)
    

    def get_random_action(self):
        return np.random.uniform(low=self.boundary_min, high=self.boundary_max, size=(self.act_dim))
    
    def load_weights(self,path,mode="all",eval=True):
        if mode == "all":
            # policy network
            self.ac.pi.load_state_dict(torch.load(path+"_pi"))
            self.ac.pi.to(self.device)
            if eval: self.ac.pi.eval()

            # Q networks
            self.ac.q.load_state_dict(torch.load(path+"_q"))
            self.ac.q.to(self.device)
            if eval: self.ac.q.eval()

            # Target networks
            self.ac_targ.pi.load_state_dict(torch.load(path+"_targ_ppi"))
            self.ac_targ.pi.to(self.device)
            if eval: self.ac_targ.pi.eval()

            self.ac_targ.q.load_state_dict(torch.load(path+"_targ_q"))
            self.ac_targ.q.to(self.device)
            if eval: self.ac_targ.q.eval()

            for p in self.ac_targ.parameters():
                p.requires_grad = False
            
             # Optimizers
            self.pi_optimizer.load_state_dict(torch.load(path+"_pi_optim"))
            for g in self.pi_optimizer.param_groups:
                g['lr'] = self.pi_lr

            self.q_optimizer.load_state_dict(torch.load(path+"_q_optim"))
            for g in self.q_optimizer.param_groups:
                g['lr'] = self.q_lr

        elif mode == "pi":
            self.ac.pi.load_state_dict(torch.load(path+"_pi"))
            self.ac.pi.to(self.device)
            if eval: self.ac.pi.eval()
    
    def save_model(self,model_path,mode="all"):
        if mode == "all":
            torch.save(self.ac.pi.state_dict(), model_path+"_pi")
            torch.save(self.ac.q.state_dict(), model_path+"_q")    
            torch.save(self.ac_targ.pi.state_dict(), model_path+"_targ_pi")   
            torch.save(self.ac_targ.q.state_dict(), model_path+"_targ_q")
            torch.save(self.pi_optimizer.state_dict(), model_path+"_pi_optim")
            torch.save(self.q_optimizer.state_dict(), model_path+"_q_optim")
        elif mode == "pi":
            torch.save(self.ac.pi.state_dict(), model_path+"_pi")
        elif mode == "q":
            torch.save(self.ac.q.state_dict(), model_path+"_q")

    
    def get_params(self):
        #print([self.ac.parameters(), self.ac_targ.parameters()])

        ac_params = []
        for p in self.ac.parameters():
            ac_params.append(p)
        
        ac_targ_params = []
        for p in self.ac_targ.parameters():
            ac_targ_params.append(p)

        pi_optim_params = []
        for p in self.pi_optimizer.param_groups:
            pi_optim_params.append(p)
        
        q_optim_params = []
        for p in self.q_optimizer.param_groups:
            q_optim_params.append(p)
        
        return [ac_params, ac_targ_params,pi_optim_params,q_optim_params]

    