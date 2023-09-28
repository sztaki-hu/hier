import numpy as np
import torch
from torch.optim import Adam
from copy import deepcopy
import itertools

import rltrain.agents.sac.core as core

class Agent:
    def __init__(self,device,config):

        self.device = device
        self.actor_critic=core.MLPActorCritic
        self.seed = config['general']['seed'] 
        self.gamma = config['agent']['gamma'] 
        
        self.polyak = config['agent']['polyak'] 
        self.lr = config['agent']['learning_rate']
        self.alpha = config['agent']['sac']['alpha']

        self.obs_dim = config['environment']['obs_dim']
        self.act_dim = config['environment']['act_dim']
        self.boundary_min = np.array(config['agent']['boundary_min'])[:self.act_dim]
        self.boundary_max = np.array(config['agent']['boundary_max'])[:self.act_dim]
        
        # torch.manual_seed(self.seed)
        # np.random.seed(self.seed)

        # # Action limit for clamping: critically, assumes all dimensions share the same bound!
        # act_limit = env.action_space.high[0]

        self.act_offset = (self.boundary_min + self.boundary_max) / 2.0
        self.act_limit = (self.boundary_max - self.boundary_min) / 2.0

        self.act_offset = torch.from_numpy(self.act_offset).float().to(self.device)
        self.act_limit = torch.from_numpy(self.act_limit).float().to(self.device)
        
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
        self.q_params = itertools.chain(self.ac.q1.parameters(), self.ac.q2.parameters())

        # Count variables (protip: try to get a feel for how different size networks behave!)
        self.var_counts = tuple(core.count_vars(module) for module in [self.ac.pi, self.ac.q1, self.ac.q2])

        # Set up optimizers for policy and q-function
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=self.lr)
        self.q_optimizer = Adam(self.q_params, lr=self.lr)

        # # Print optimizer's state_dict
        # print("Optimizer's state_dict:")
        # for var_name in self.pi_optimizer.state_dict():
        #     print(var_name, "\t", self.pi_optimizer.state_dict()[var_name])

        if config['general']['init_weights']['bool']:
            self.init_weights_path = config['general']['init_weights']['path']
            self.init_weights_mode = config['general']['init_weights']['mode']
            self.load_weights(self.init_weights_path, mode=self.init_weights_mode, eval=False)

        """
        Soft Actor-Critic (SAC)

        Args:
            actor_critic: The constructor method for a PyTorch Module with an ``act`` 
                method, a ``pi`` module, a ``q1`` module, and a ``q2`` module.
                The ``act`` method and ``pi`` module should accept batches of 
                observations as inputs, and ``q1`` and ``q2`` should accept a batch 
                of observations and a batch of actions as inputs. When called, 
                ``act``, ``q1``, and ``q2`` should return:

                ===========  ================  ======================================
                Call         Output Shape      Description
                ===========  ================  ======================================
                ``act``      (batch, act_dim)  | Numpy array of actions for each 
                                            | observation.
                ``q1``       (batch,)          | Tensor containing one current estimate
                                            | of Q* for the provided observations
                                            | and actions. (Critical: make sure to
                                            | flatten this!)
                ``q2``       (batch,)          | Tensor containing the other current 
                                            | estimate of Q* for the provided observations
                                            | and actions. (Critical: make sure to
                                            | flatten this!)
                ===========  ================  ======================================

                Calling ``pi`` should return:

                ===========  ================  ======================================
                Symbol       Shape             Description
                ===========  ================  ======================================
                ``a``        (batch, act_dim)  | Tensor containing actions from policy
                                            | given observations.
                ``logp_pi``  (batch,)          | Tensor containing log probabilities of
                                            | actions in ``a``. Importantly: gradients
                                            | should be able to flow back into ``a``.
                ===========  ================  ======================================

            ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object 
                you provided to SAC.

            seed (int): Seed for random number generators.

            gamma (float): Discount factor. (Always between 0 and 1.)

            polyak (float): Interpolation factor in polyak averaging for target 
                networks. Target networks are updated towards main networks 
                according to:

                .. math:: \\theta_{\\text{targ}} \\leftarrow 
                    \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

                where :math:`\\rho` is polyak. (Always between 0 and 1, usually 
                close to 1.)

            lr (float): Learning rate (used for both policy and value learning).

            alpha (float): Entropy regularization coefficient. (Equivalent to 
                inverse of reward scale in the original SAC paper.)
        """

    # Set up function for computing SAC Q-losses
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
            # Target actions come from *current* policy
            a2, logp_a2 = self.ac.pi(o2)

            # Target Q-values
            q1_pi_targ = self.ac_targ.q1(o2, a2)
            q2_pi_targ = self.ac_targ.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + self.gamma * (1 - d) * (q_pi_targ - self.alpha * logp_a2)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        q_info = dict(Q1Vals=q1.cpu().detach().numpy(),
                      Q2Vals=q2.cpu().detach().numpy())

        return loss_q, q_info

    # Set up function for computing SAC pi loss
    def compute_loss_pi(self,data):

        o = data['obs'].to(self.device)

        pi, logp_pi = self.ac.pi(o)

        q1_pi = self.ac.q1(o, pi)
        q2_pi = self.ac.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = (self.alpha * logp_pi - q_pi).mean()

        # Useful info for logging
        pi_info = dict(LogPi=logp_pi.cpu().detach().numpy())

        return loss_pi, pi_info

    def update(self, data, placeholder):

        # First run one gradient descent step for Q1 and Q2
        self.q_optimizer.zero_grad()
        loss_q, q_info = self.compute_loss_q(data)
        loss_q.backward()
        self.q_optimizer.step()

        #self.logger.tb_writer_add_scalar("train/loss_q", loss_q.item(), update_iter)
        ret_loss_q = loss_q.item()

        # Freeze Q-networks so you don't waste computational effort 
        # computing gradients for them during the policy learning step.
        for p in self.q_params:
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        self.pi_optimizer.zero_grad()
        loss_pi, pi_info = self.compute_loss_pi(data)
        loss_pi.backward()
        self.pi_optimizer.step()

        #self.logger.tb_writer_add_scalar("train/loss_pi", loss_pi.item(), update_iter)
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

    def get_action(self, o, deterministic=False):
        o = torch.from_numpy(o).float().unsqueeze(0).to(self.device)
        return self.ac.act(o, deterministic)[0]

    def get_random_action(self):
        return np.random.uniform(low=self.boundary_min, high=self.boundary_max, size=(self.act_dim))
    
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

            


    
    
        