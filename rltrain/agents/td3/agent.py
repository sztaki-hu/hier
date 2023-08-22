from copy import deepcopy
import itertools
import numpy as np
import torch
from torch.optim import Adam

import rltrain.agents.td3.core as core

class Agent:

    def __init__(self,device,config):

        self.device = device
        self.actor_critic=core.MLPActorCritic
        self.seed = config['general']['seed'] 
        self.gamma = config['agent']['gamma']
        
        self.target_noise = config['agent']['td3']['target_noise'] 
        self.noise_clip = config['agent']['td3']['noise_clip'] 
        self.pi_lr = config['agent']['td3']['pi_lr']  
        self.q_lr = config['agent']['td3']['q_lr'] 
        self.policy_delay = config['agent']['td3']['policy_delay'] 
        self.polyak = config['agent']['td3']['polyak'] 
        self.act_noise = config['agent']['td3']['act_noise'] 

        self.obs_dim = config['environment']['obs_dim']
        self.act_dim = config['environment']['act_dim']
        self.boundary_min = np.array(config['agent']['boundary_min'])[:self.act_dim]
        self.boundary_max = np.array(config['agent']['boundary_max'])[:self.act_dim]

        self.act_offset = (self.boundary_min + self.boundary_max) / 2.0
        self.act_limit = (self.boundary_max - self.boundary_min) / 2.0

        self.act_offset = torch.from_numpy(self.act_offset).float().to(self.device)
        self.act_limit = torch.from_numpy(self.act_limit).float().to(self.device)

        #self.act_limit_np = self.act_limit.detach().cpu().numpy()

        self.act_min = torch.from_numpy(self.boundary_min).float().to(self.device)
        self.act_max = torch.from_numpy(self.boundary_max).float().to(self.device)

        # Create actor-critic module and target networks
        #ac = actor_critic(obs_dim, act_dim, act_offset, act_limit)
        self.ac = self.actor_critic(self.obs_dim, self.act_dim, self.act_offset, self.act_limit, hidden_sizes=tuple(config['agent']['hidden_sizes']))
        
        self.ac.pi.to(self.device)
        self.ac.q1.to(self.device)
        self.ac.q2.to(self.device)

        print(self.ac.pi)
        print(self.ac.q1)
        print(self.ac.q2)

        self.ac_targ = deepcopy(self.ac)

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.ac_targ.parameters():
            p.requires_grad = False
            
        # List of parameters for both Q-networks (save this for convenience)
        self.q_params = itertools.chain(self.ac.q1.parameters(),  self.ac.q2.parameters())

        # Count variables (protip: try to get a feel for how different size networks behave!)
        self.var_counts = tuple(core.count_vars(module) for module in [self.ac.pi,  self.ac.q1,  self.ac.q2])

       
        # Set up optimizers for policy and q-function
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=self.pi_lr)
        self.q_optimizer = Adam(self.q_params, lr=self.q_lr)


    # Set up function for computing TD3 Q-losses
    def compute_loss_q(self, data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        o = o.float().to(self.device)
        a = a.to(self.device)
        r = r.to(self.device)
        o2 = o2.float().to(self.device)
        d = d.to(self.device)


        q1 = self.ac.q1(o,a)
        q2 = self.ac.q2(o,a)

        # Bellman backup for Q functions
        with torch.no_grad():
            pi_targ = self.ac_targ.pi(o2)

            # Target policy smoothing
            epsilon = torch.randn_like(pi_targ) * self.target_noise
            epsilon = torch.clamp(epsilon, -self.noise_clip, self.noise_clip)
            a2 = pi_targ + epsilon
            
            #a2 = torch.clamp(a2, -self.act_limit_np, self.act_limit_np)
            a2 = torch.max(torch.min(a2, self.act_max), self.act_min)

            # Target Q-values
            q1_pi_targ = self.ac_targ.q1(o2, a2)
            q2_pi_targ = self.ac_targ.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + self.gamma * (1 - d) * q_pi_targ

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        loss_info = dict(Q1Vals=q1.cpu().detach().numpy(),
                        Q2Vals=q2.cpu().detach().numpy())

        return loss_q, loss_info

    # Set up function for computing TD3 pi loss
    def compute_loss_pi(self,data):

        o = data['obs'].to(self.device)

        q1_pi = self.ac.q1(o, self.ac.pi(o))
        return -q1_pi.mean()


    def update(self,data, timer):
        # First run one gradient descent step for Q1 and Q2
        self.q_optimizer.zero_grad()
        loss_q, loss_info = self.compute_loss_q(data)
        loss_q.backward()
        self.q_optimizer.step()

        ret_loss_q = loss_q.item()

        ret_loss_pi = None

        # Possibly update pi and target networks
        if timer % self.policy_delay == 0:

            # Freeze Q-networks so you don't waste computational effort 
            # computing gradients for them during the policy learning step.
            for p in self.q_params:
                p.requires_grad = False

            # Next run one gradient descent step for pi.
            self.pi_optimizer.zero_grad()
            loss_pi = self.compute_loss_pi(data)
            loss_pi.backward()
            self.pi_optimizer.step()

            ret_loss_pi = loss_pi.item()

            # Unfreeze Q-networks so you can optimize it at next DDPG step.
            for p in self.q_params:
                p.requires_grad = True

            # Finally, update target networks by polyak averaging.
            with torch.no_grad():
                for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                    # NB: We use an in-place operations "mul_", "add_" to update target
                    # params, as opposed to "mul" and "add", which would make new tensors.
                    p_targ.data.mul_(self.polyak)
                    p_targ.data.add_((1 - self.polyak) * p.data)
            
        return ret_loss_q, ret_loss_pi
    
    def get_action(self,o, noise_scale):
        o = torch.from_numpy(o).float().unsqueeze(0).to(self.device)
        a = self.ac.act(o)[0]
        a += noise_scale * np.random.randn(self.act_dim)
        return np.clip(a, self.boundary_min, self.boundary_max)

    # def get_action(self, o, deterministic=False):
    #     o = torch.from_numpy(o).float().unsqueeze(0).to(self.device)
    #     return self.ac.act(o, deterministic)[0]

    def get_random_action(self):
        return np.random.uniform(low=self.boundary_min, high=self.boundary_max, size=(self.act_dim))

    def save_model(self,model_path,mode="all"):
        if mode == "all":
            torch.save(self.ac.pi.state_dict(), model_path+"_pi")
            torch.save(self.ac.q1.state_dict(), model_path+"_q1")
            torch.save(self.ac.q2.state_dict(), model_path+"_q2")        
            torch.save(self.ac_targ.pi.state_dict(), model_path+"_targ_ppi")   
            torch.save(self.ac_targ.q1.state_dict(), model_path+"_targ_q1")
            torch.save(self.ac_targ.q2.state_dict(), model_path+"_targ_q2")
            torch.save(self.pi_optimizer.state_dict(), model_path+"_pi_optim")
            torch.save(self.q_optimizer.state_dict(), model_path+"_q_optim")
        elif mode == "pi":
            torch.save(self.ac.pi.state_dict(), model_path+"_pi")
        elif mode == "q":
            torch.save(self.ac.q1.state_dict(), model_path+"_q1")
            torch.save(self.ac.q2.state_dict(), model_path+"_q2")
    
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

    def load_weights(self,path,mode="all",eval=True):
        if mode == "all":
            # policy network
            self.ac.pi.load_state_dict(torch.load(path+"_pi"))
            self.ac.pi.to(self.device)
            if eval: self.ac.pi.eval()

            # Q networks
            self.ac.q1.load_state_dict(torch.load(path+"_q1"))
            self.ac.q1.to(self.device)
            if eval: self.ac.q1.eval()

            self.ac.q2.load_state_dict(torch.load(path+"_q2"))
            self.ac.q2.to(self.device)
            if eval: self.ac.q2.eval()

            # Target networks
            self.ac_targ.pi.load_state_dict(torch.load(path+"_targ_ppi"))
            self.ac_targ.pi.to(self.device)
            if eval: self.ac_targ.pi.eval()

            self.ac_targ.q1.load_state_dict(torch.load(path+"_targ_q1"))
            self.ac_targ.q1.to(self.device)
            if eval: self.ac_targ.q1.eval()

            self.ac_targ.q2.load_state_dict(torch.load(path+"_targ_q2"))
            self.ac_targ.q2.to(self.device)
            if eval: self.ac_targ.q2.eval()

            for p in self.ac_targ.parameters():
                p.requires_grad = False
            
             # Optimizers
            self.pi_optimizer.load_state_dict(torch.load(path+"_pi_optim"))
            for g in self.pi_optimizer.param_groups:
                g['lr'] = self.lr

            self.q_optimizer.load_state_dict(torch.load(path+"_q_optim"))
            for g in self.q_optimizer.param_groups:
                g['lr'] = self.lr

        elif mode == "pi":
            self.ac.pi.load_state_dict(torch.load(path+"_pi"))
            self.ac.pi.to(self.device)
            if eval: self.ac.pi.eval()
