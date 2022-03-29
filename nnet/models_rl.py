
# Copyright 2021, Maxime Burchi.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# PyTorch
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

# NeuralNets
from nnet.model import Model
from nnet.models import (
    JoinedDDPG
)
from nnet.buffers import (
    ReplayBuffer,
    PrioritizedReplayBuffer
)
from nnet.losses import (
    MeanLoss,
    MeanSquaredError,
    NegCosSim,
    KullbackLeiblerDivergence
)
from nnet.collate_fn import (
    CollateList
)
from nnet.optimizers import (
    Adam
)
from nnet.noises import (
    OrnsteinUhlenbeckProcess
)

# Other
#import matplotlib.pyplot as plt
import copy

class MuZero(nn.Module):

    """ MuZero

    Default Atari params
    
    Reference: "Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model", Schrittwieser et al.
    https://arxiv.org/abs/1911.08265

    """

    def __init__(self, h_net, f_net, g_net, env, K=5, N=10, num_sim=50, gamma=0.997, batch_size=1024, buffer_size=125000, buffer_alpha=1, buffer_beta=1, eps_scaling=0.001, search_c1=1.25, search_c2=19652.0):
        super(MuZero, self).__init__()

        # Representation Network
        self.h_net = h_net

        # Prediction Network
        self.f_net = f_net

        # Dynamics Network
        self.g_net = g_net

        # Env
        self.env = env

        # replay Buffer
        self.buffer = PrioritizedReplayBuffer(
            size=buffer_size,
            state_size=self.env.state_size,
            action_size=self.env.action_size,
            alpha=buffer_alpha,
            beta=buffer_beta
        )

        # Params
        self.K = K
        self.N = N
        self.num_sim = num_sim
        self.gamma = gamma
        self.batch_size = batch_size
        self.search_c1 = search_c1
        self.search_c2 = search_c2

    def compile(
        self, 
        optimizer="Adam",
        losses={"reward": MeanSquaredError(), "value": MeanSquaredError(), "policy": MeanSquaredError()},
        loss_weights={"reward": 1.0, "value": 1.0, "policy": 1.0},
        metrics=None,
        decoders=None,
        collate_fn=CollateList(inputs_axis=[], targets_axis=[])
    ):
        # Compile Model
        super(MuZero, self).compile(
            optimizer=Adam(params=[{"params": self.h_net.parameters()}, {"params": self.f_net.parameters()}, {"params": self.g_net.parameters()}], lr=0.001) if optimizer == "Adam" else optimizer,
            losses=losses,
            loss_weights=loss_weights,
            metrics=metrics,
            decoders=decoders,
            collate_fn=collate_fn
        )

    def summary(self, show_dict=False):
        super(MuZero, self).summary()

        print("Representation Network Parameters: {:,}".format(self.num_params(self.h_net)))
        if show_dict:
            self.show_dict(self.h_net)

        print("Prediction Network Parameters: {:,}".format(self.num_params(self.f_net)))
        if show_dict:
            self.show_dict(self.f_net)

        print("Dynamics Network Parameters: {:,}".format(self.num_params(self.g_net)))
        if show_dict:
            self.show_dict(self.g_net)

        print("Replay Buffer:")
        if show_dict:
            self.show_dict(self.buffer)

    def env_step(self):

        # Get State
        state = self.state.to(self.device)

        # Forward Policy Network
        with torch.no_grad():
            self.eval()
            policy, value = self.MCTS(state.unsqueeze(dim=0))
            self.train()

        # Action Info
        self.infos["actions"] = ["{}{:.2f}".format("+" if a >= 0 else "-", abs(a)) for a in action.squeeze(dim=0).tolist()]

        # Update Step
        self.action_step += 1

        # Add Noise
        action += self.noise()

        # Clip Action
        action = action.clip(self.clip_low, self.clip_high)

        # Env Step
        state_next, reward, done = self.env.step(action.squeeze(dim=0))

        # Render
        #self.env.env.render()

        # Store Transitions
        if not done or self.include_done_transition:
            self.buffer.append(state, action.squeeze(dim=0), torch.tensor(reward, device=self.device, dtype=torch.float32), state_next.to(self.device), torch.tensor(done, device=self.device, dtype=torch.float32))

        # Update ep rewards
        self.ep_rewards += reward

        # Done
        if done:
            self.state = self.env.reset()
            self.episodes += 1
            self.running_rewards = 0.05 * self.ep_rewards + (1 - 0.05) * self.running_rewards
            self.ep_rewards = 0.0
            self.noise.reset()
        else:
            self.state = state_next

    def MCTS(self):
        pass




class PrioritizedJDDPG(JoinedDDPG):

    def __init__(self, r_net, p_net, q_net, env, batch_size=64, gamma=0.99, tau=0.001, noise_theta=0.15, noise_std=0.2, noise_dt=0.05, buffer_size=50000, update_period=1, reward_done=None, include_done_transition=True, name="Prioritized Joined Deep Deterministic Policy Gradient Model"):
        super(PrioritizedJDDPG, self).__init__(r_net=r_net, p_net=p_net, q_net=q_net, env=env, batch_size=batch_size, gamma=gamma, tau=tau, noise_theta=noise_theta, noise_std=noise_std, noise_dt=noise_dt, buffer_size=buffer_size, update_period=update_period, reward_done=reward_done, include_done_transition=include_done_transition, name=name)

        # Replay Buffer
        self.buffer = PrioritizedReplayBuffer(
            size=buffer_size,
            state_size=self.env.state_size,
            action_size=self.env.action_size
        )

    def compile(
        self, 
        optimizer="Adam",
        losses=[MeanLoss(targets_as_sign=False), MeanLoss(targets_as_sign=False)],
        loss_weights=[0.1, 1.0],
        metrics=None,
        decoders=None,
        collate_fn=CollateList(inputs_axis=[], targets_axis=[])
    ):
        # Compile Model
        super(PrioritizedJDDPG, self).compile(
            optimizer=Adam(params=[{"params": self.r_net.parameters()}, {"params": self.p_net.parameters()}, {"params": self.q_net.parameters()}], lr=0.001) if optimizer == "Adam" else optimizer,
            losses=losses,
            loss_weights=loss_weights,
            metrics=metrics,
            decoders=decoders,
            collate_fn=collate_fn
        )

    def forward(self, inputs):

        # Unpack Inputs
        states, actions, rewards, states_next, dones, indices, weights = inputs

        # Forward Representation Network
        r_states = self.r_net(states)

        # Forward Policy Network
        pred_actions = self.p_net(r_states)

        # Forward Q-Value Network
        actor_pred_returns = self.q_frozen([r_states, pred_actions])

        # Forward Q-Value Network
        critic_pred_returns = self.q_net([r_states, actions])

        # Compute Q-Value Network Targets
        with torch.no_grad():
            r_states_next = self.r_target(states_next)
            if self.reward_done != None:
                critic_targets = (rewards.unsqueeze(-1) + self.gamma * self.q_target([r_states_next, self.p_target(r_states_next)])) * (1 - dones.unsqueeze(-1)) + self.reward_done * dones.unsqueeze(-1)
            else:
                critic_targets = rewards.unsqueeze(-1) + self.gamma * self.q_target([r_states_next, self.p_target(r_states_next)]) * (1 - dones.unsqueeze(-1)) 

        # Compute Sigmas
        sigmas = (critic_targets - critic_pred_returns).squeeze(dim=-1)

        # Compute Critic Loss
        loss_critic = 0.5 * sigmas.square()
        #loss_critic = F.mse_loss(critic_pred_returns, critic_targets, reduction="none").squeeze(dim=-1)

        # Update Sigmas
        self.buffer.update_sigmas(sigmas.abs().detach(), indices)

        return {"actor": - weights * actor_pred_returns, "critic": weights * loss_critic}


class ModelJDDPG(JoinedDDPG):

    def __init__(self, r_net, p_net, q_net, m_net, env, batch_size=64, gamma=0.99, tau=0.001, noise_theta=0.15, noise_std=0.2, noise_dt=0.05, buffer_size=50000, update_period=1, reward_done=None, include_done_transition=True, name="Model Based Joined Deep Deterministic Policy Gradient Model"):
        super(ModelJDDPG, self).__init__(r_net=r_net, p_net=p_net, q_net=q_net, env=env, batch_size=batch_size, gamma=gamma, tau=tau, noise_theta=noise_theta, noise_std=noise_std, noise_dt=noise_dt, buffer_size=buffer_size, update_period=update_period, reward_done=reward_done, include_done_transition=include_done_transition, name=name)

        # Model Networks
        self.m_net = m_net
        self.m_target = type(self.m_net)()
        self.m_target.load_state_dict(self.m_net.state_dict())
        self.set_require_grad(self.m_target, False)

    def train_step(self, inputs, targets, mixed_precision, grad_scaler, accumulated_steps, acc_step):
        batch_losses, batch_metrics, _ = super(ModelJDDPG, self).train_step(inputs=inputs, targets=targets, mixed_precision=mixed_precision, grad_scaler=grad_scaler, accumulated_steps=accumulated_steps, acc_step=acc_step)

        # Update Target Network
        for param_target, param_net in zip(self.m_target.parameters(), self.m_net.parameters()):
            param_target.mul_(1 - self.tau)
            param_target.add_(self.tau * param_net.detach())

        return batch_losses, batch_metrics, _

    def compile(
        self, 
        optimizer="Adam",
        losses=[MeanLoss(targets_as_sign=False), MeanSquaredError(), MeanSquaredError(), MeanSquaredError(), MeanSquaredError()],
        loss_weights=[0.1, 1.0, 1.0, 1.0, 1.0],
        metrics=None,
        decoders=None,
        collate_fn=CollateList(inputs_axis=[], targets_axis=[])
    ):
        # Compile Model
        super(ModelJDDPG, self).compile(
            optimizer=Adam(params=[{"params": self.r_net.parameters()}, {"params": self.p_net.parameters()}, {"params": self.q_net.parameters()}, {"params": self.m_net.parameters()}], lr=0.001) if optimizer == "Adam" else optimizer,
            losses=losses,
            loss_weights=loss_weights,
            metrics=metrics,
            decoders=decoders,
            collate_fn=collate_fn
        )

    def summary(self, show_dict=False): 
        super(ModelJDDPG, self).summary(show_dict=show_dict)

        print("Model Network Parameters:", self.num_params(self.m_net))
        if show_dict:
            self.show_dict(self.m_net)

    def forward(self, inputs):

        # Unpack Inputs
        states, actions, rewards, states_next, dones = inputs

        # Forward Representation Network
        r_states = self.r_net(states)

        # Forward Policy Network
        pred_actions = self.p_net(r_states)

        # Forward Q-Value Network
        actor_pred_returns = self.q_frozen([r_states, pred_actions])

        # Forward Q-Value Network
        critic_pred_returns = self.q_net([r_states, actions])

        # Compute Q-Value Network Targets
        with torch.no_grad():
            r_states_next_img, rewards_img = self.m_target([r_states, actions])
            self.additional_targets["critic"] = rewards_img + self.gamma * self.q_target([r_states_next_img, self.p_target(r_states_next_img)])
            
        # Forward Model
        r_states_next_img, rewards_img = self.m_net([r_states, actions])

        # Forward Policy network
        pred_actions_img = self.p_net(r_states_next_img)

        # Forward Q-Value Network
        model_pred_returns = self.q_net([r_states_next_img, pred_actions_img])

        # Compute Model Network Targets
        with torch.no_grad():

            # Real Targets
            r_states_next = self.r_target(states_next)
            actions_next = self.p_target(r_states_next)
            returns = rewards.unsqueeze(-1) + self.gamma * self.q_target([r_states_next, actions_next])

            # Rewards Targets
            self.additional_targets["model_r"] = rewards.unsqueeze(-1)

            # Returns Targets
            self.additional_targets["model_a"] = actions_next

            # Returns Targets
            self.additional_targets["model_R"] = returns

        return {"actor": - actor_pred_returns, "critic": critic_pred_returns, "model_r": rewards_img, "model_a": pred_actions_img, "model_R": model_pred_returns}

class ModelJDDPG2(JoinedDDPG):

    def __init__(self, r_net, p_net, q_net, m_net, proj_net, pred_net, env, batch_size=64, gamma=0.99, tau=0.001, noise_theta=0.15, noise_std=0.2, noise_dt=0.05, buffer_size=50000, update_period=1, reward_done=None, include_done_transition=True, name="Model Based Joined Deep Deterministic Policy Gradient Model"):
        super(ModelJDDPG2, self).__init__(r_net=r_net, p_net=p_net, q_net=q_net, env=env, batch_size=batch_size, gamma=gamma, tau=tau, noise_theta=noise_theta, noise_std=noise_std, noise_dt=noise_dt, buffer_size=buffer_size, update_period=update_period, reward_done=reward_done, include_done_transition=include_done_transition, name=name)

        # Model Networks
        self.m_net = m_net
        self.m_target = type(self.m_net)()
        self.m_target.load_state_dict(self.m_net.state_dict())
        self.set_require_grad(self.m_target, False)

        # Proj Networks
        self.proj_net = proj_net
        self.proj_target = type(self.proj_net)()
        self.proj_target.load_state_dict(self.proj_net.state_dict())
        self.set_require_grad(self.proj_target, False)

        # Pred Network
        self.pred_net = pred_net

    def train_step(self, inputs, targets, mixed_precision, grad_scaler, accumulated_steps, acc_step):
        batch_losses, batch_metrics, _ = super(ModelJDDPG2, self).train_step(inputs=inputs, targets=targets, mixed_precision=mixed_precision, grad_scaler=grad_scaler, accumulated_steps=accumulated_steps, acc_step=acc_step)

        # Update Target Network
        for param_target, param_net in zip(self.m_target.parameters(), self.m_net.parameters()):
            param_target.mul_(1 - self.tau)
            param_target.add_(self.tau * param_net.detach())
        for param_target, param_net in zip(self.proj_target.parameters(), self.proj_net.parameters()):
            param_target.mul_(1 - self.tau)
            param_target.add_(self.tau * param_net.detach())

        return batch_losses, batch_metrics, _

    def compile(
        self, 
        optimizer="Adam",
        losses={"actor": MeanLoss(targets_as_sign=False), "critic": MeanSquaredError(), "model_s": NegCosSim(), "model_r": MeanSquaredError(), "model_a": MeanSquaredError(), "model_R": MeanSquaredError()},
        loss_weights={"actor": 0.1, "critic": 1.0, "model_s": 1.0, "model_r": 1.0, "model_a": 1.0, "model_R": 1.0},
        metrics=None,
        decoders=None,
        collate_fn=CollateList(inputs_axis=[], targets_axis=[])
    ):
        # Compile Model
        super(ModelJDDPG2, self).compile(
            optimizer=Adam(params=[{"params": self.r_net.parameters()}, {"params": self.p_net.parameters()}, {"params": self.q_net.parameters()}, {"params": self.m_net.parameters()}], lr=0.001) if optimizer == "Adam" else optimizer,
            losses=losses,
            loss_weights=loss_weights,
            metrics=metrics,
            decoders=decoders,
            collate_fn=collate_fn
        )

    def summary(self, show_dict=False): 
        super(ModelJDDPG2, self).summary(show_dict=show_dict)

        print("Model Network Parameters: {:,}".format(self.num_params(self.m_net)))
        if show_dict:
            self.show_dict(self.m_net)

        print("Projection Network Parameters: {:,}".format(self.num_params(self.proj_net)))
        if show_dict:
            self.show_dict(self.proj_net)

        print("Prediction Network Parameters: {:,}".format(self.num_params(self.pred_net)))
        if show_dict:
            self.show_dict(self.pred_net)

    def forward(self, inputs):

        # Unpack Inputs
        states, actions, rewards, states_next, dones = inputs

        ###############################################################################
        # Model Step
        ###############################################################################

        # Forward Representation Network
        r_states = self.r_net(states)

        # Forward Model
        model_r_states_next, model_rewards_pred = self.m_net([r_states, actions])
        model_r_states_next_pred = self.pred_net(self.proj_net(model_r_states_next))

        # Forward Policy network
        model_actions_pred = self.p_net(model_r_states_next)

        # Forward Q-Value Network
        model_returns_pred = self.q_net([model_r_states_next, model_actions_pred])

        # Compute Model Network Targets
        with torch.no_grad():

            # Real Targets
            r_states_next = self.r_target(states_next)
            actions_next = self.p_target(r_states_next)
            rewards = rewards.unsqueeze(-1)

            # Returns Targets
            self.additional_targets["model_s"] = self.proj_target(r_states_next)

            # Rewards Targets
            self.additional_targets["model_r"] = rewards

            # Returns Targets
            self.additional_targets["model_a"] = actions_next

            # Returns Targets
            self.additional_targets["model_R"] = rewards + self.gamma * self.q_target([r_states_next, actions_next])

        ###############################################################################
        # Imaginary Step
        ###############################################################################

        # Forward Policy Network
        pred_actions = self.p_net(r_states)

        # Forward Q-Value Network
        actor_pred_returns = self.q_frozen([r_states, pred_actions])

        # Forward Q-Value Network
        critic_pred_returns = self.q_net([r_states, actions])

        # Compute Q-Value Network Targets
        with torch.no_grad():
            r_states_next_img, rewards_img = self.m_target([r_states, actions])
            self.additional_targets["critic"] = rewards_img + self.gamma * self.q_target([r_states_next_img, self.p_target(r_states_next_img)])

        return {"actor": - actor_pred_returns, "critic": critic_pred_returns, "model_s": model_r_states_next_pred, "model_r": model_rewards_pred, "model_a": model_actions_pred, "model_R": model_returns_pred}

class ImgJDDPG(ModelJDDPG):

    def __init__(self, r_net, p_net, q_net, m_net, e_net, d_net, env, batch_size=64, gamma=0.99, tau=0.001, noise_theta=0.15, noise_std=0.2, noise_dt=0.05, buffer_size=50000, update_period=1, reward_done=None, include_done_transition=True, name="Imaginary Joined Deep Deterministic Policy Gradient Model"):
        super(ImgJDDPG, self).__init__(r_net=r_net, p_net=p_net, q_net=q_net, m_net=m_net, env=env, batch_size=batch_size, gamma=gamma, tau=tau, noise_theta=noise_theta, noise_std=noise_std, noise_dt=noise_dt, buffer_size=buffer_size, update_period=update_period, reward_done=reward_done, include_done_transition=include_done_transition, name=name)

        # VAE Networks
        self.e_net = e_net
        self.d_net = d_net

    def compile(
        self, 
        optimizer="Adam",
        losses={"actor": MeanLoss(targets_as_sign=False), "critic": MeanSquaredError(), "model_r": MeanSquaredError(), "model_a": MeanSquaredError(), "model_R": MeanSquaredError(), "vae": MeanSquaredError(), "KLD": KullbackLeiblerDivergence()},
        loss_weights={"actor": 0.1, "critic": 1.0, "model_r": 1.0, "model_a": 1.0, "model_R": 1.0, "vae": 1.0, "KLD": 0.01},
        metrics=None,
        decoders=None,
        collate_fn=CollateList(inputs_axis=[], targets_axis=[])
    ):
        # Compile Model
        super(ImgJDDPG, self).compile(
            optimizer=Adam(params=[{"params": self.r_net.parameters()}, {"params": self.p_net.parameters()}, {"params": self.q_net.parameters()}, {"params": self.m_net.parameters()}, {"params": self.e_net.parameters()}, {"params": self.d_net.parameters()}], lr=0.001) if optimizer == "Adam" else optimizer,
            losses=losses,
            loss_weights=loss_weights,
            metrics=metrics,
            decoders=decoders,
            collate_fn=collate_fn
        )

    def summary(self, show_dict=False): 
        super(ImgJDDPG, self).summary(show_dict=show_dict)

        print("Encoder Network Parameters:", self.num_params(self.e_net))
        if show_dict:
            self.show_dict(self.e_net)

        print("Decoder Network Parameters:", self.num_params(self.d_net))
        if show_dict:
            self.show_dict(self.d_net)

    def forward(self, inputs):

        # Unpack Inputs
        states, actions, rewards, states_next, dones = inputs

        ###############################################################################
        # Model Step
        ###############################################################################

        # Forward Representation network
        r_states = self.r_net(states)
            
        # Forward Model
        model_r_states_next, model_rewards_pred = self.m_net([r_states, actions])

        # Forward Policy network
        model_actions_pred = self.p_net(model_r_states_next)

        # Forward Q-Value Network
        model_returns_pred = self.q_net([model_r_states_next, model_actions_pred])

        # Compute Model Network Targets
        with torch.no_grad():

            # Real Targets
            r_states_next = self.r_target(states_next)
            actions_next = self.p_target(r_states_next)
            rewards = rewards.unsqueeze(-1)

            # Rewards Targets
            self.additional_targets["model_r"] = rewards

            # Returns Targets
            self.additional_targets["model_a"] = actions_next

            # Returns Targets
            self.additional_targets["model_R"] = rewards + self.gamma * self.q_net([r_states_next, actions_next])

        ###############################################################################
        # VAE Step
        ###############################################################################

        # Forward Encoder Network
        mean, log_var = self.e_net(r_states)

        # Reparametrization
        vae_r_states_latent = torch.randn(size=mean.size(), device=mean.device) * torch.exp(log_var * 0.5) + mean

        # Forward Decoder Network
        vae_r_states = self.d_net(vae_r_states_latent)

        # VAE Network Targets
        with torch.no_grad():
            self.additional_targets["vae"] = r_states

        ###############################################################################
        # Imaginary Step
        ###############################################################################

        # Generate states
        r_states = self.d_net(torch.randn(size=(self.batch_size,) + mean.shape[1:], device=mean.device)).detach()

        # Forward Policy Network
        pred_actions = self.p_net(r_states)

        # Forward Q-Value Network
        actor_pred_returns = self.q_frozen([r_states, pred_actions])

        # Add Noise to Actions
        actions = pred_actions.detach()#(pred_actions + torch.normal(mean=0, std=0.2, size=actions.size(), device=pred_actions.device)).detach()
        
        # Clip Action
        #actions = actions.clip(self.clip_low, self.clip_high)

        # Forward Q-Value Network
        critic_pred_returns = self.q_net([r_states, actions])

        # Compute Q-Value Network Targets
        with torch.no_grad():
            r_states_next_img, rewards_img = self.m_target([r_states, actions])
            self.additional_targets["critic"] = rewards_img + self.gamma * self.q_target([r_states_next_img, self.p_target(r_states_next_img)])

        return {"actor": - actor_pred_returns, "critic": critic_pred_returns, "model_r": model_rewards_pred, "model_a": model_actions_pred, "model_R": model_returns_pred, "vae": vae_r_states, "KLD": [mean, log_var]}

class OnlineImgJDDPG(Model):

    def __init__(self, r_net, p_net, q_net, m_net, e_net, d_net, env, batch_size=64, gamma=0.99, tau=0.001, noise_theta=0.15, noise_std=0.2, noise_dt=0.05, update_period=1, reward_done=None, include_done_transition=True, name="Imaginary Joined Deep Deterministic Policy Gradient Model"):
        super(OnlineImgJDDPG, self).__init__(name=name)

        # Representation Networks
        self.r_net = r_net
        self.r_target = type(self.r_net)()
        self.r_target.load_state_dict(self.r_net.state_dict())
        self.set_require_grad(self.r_target, False)

        # Policy Networks
        self.p_net = p_net
        self.p_target = type(self.p_net)()
        self.p_target.load_state_dict(self.p_net.state_dict())
        self.set_require_grad(self.p_target, False)

        # Q-Value Networks
        self.q_net = q_net
        self.q_target = type(self.q_net)()
        self.q_target.load_state_dict(self.q_net.state_dict())
        self.set_require_grad(self.q_target, False)
        self.q_frozen = type(self.q_net)()
        self.q_frozen.load_state_dict(self.q_net.state_dict())
        self.set_require_grad(self.q_frozen, False)

        # Model Networks
        self.m_net = m_net
        self.m_target = type(self.m_net)()
        self.m_target.load_state_dict(self.m_net.state_dict())
        self.set_require_grad(self.m_target, False)

        # VAE Networks
        self.e_net = e_net
        self.d_net = d_net

        # Env
        self.env = env
        self.state = self.env.reset()
        self.clip_low = self.env.env.action_space.low[0]
        self.clip_high = self.env.env.action_space.high[0]

        # Noise Module
        self.noise = OrnsteinUhlenbeckProcess(num_actions=self.env.num_actions, mean=0, std=noise_std, theta=noise_theta, dt=noise_dt)

        # Training Params
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.update_period = update_period
        self.reward_done = reward_done
        self.include_done_transition = include_done_transition
        self.noise_std = noise_std
        self.noise_dt = noise_dt

        # Training Infos
        self.episodes = 0
        self.running_rewards = 0.0
        self.ep_rewards = 0.0
        self.action_step = 0

        # Networks
        self.networks = [self.r_net, self.p_net, self.q_net, self.m_net, self.e_net, self.d_net]

    def compile(
        self, 
        optimizer="Adam",
        losses={"actor": MeanLoss(targets_as_sign=False), "critic": MeanSquaredError(), "model_r": MeanSquaredError(), "model_a": MeanSquaredError(), "model_R": MeanSquaredError(), "vae": MeanSquaredError(), "KLD": KullbackLeiblerDivergence()},
        loss_weights={"actor": 0.1, "critic": 1.0, "model_r": 1.0, "model_a": 1.0, "model_R": 1.0, "vae": 1.0, "KLD": 0.01},
        metrics=None,
        decoders=None,
        collate_fn=CollateList(inputs_axis=[], targets_axis=[])
    ):
        # Compile Model
        super(OnlineImgJDDPG, self).compile(
            optimizer=Adam(params=[{"params": self.r_net.parameters()}, {"params": self.p_net.parameters()}, {"params": self.q_net.parameters()}, {"params": self.m_net.parameters()}, {"params": self.e_net.parameters()}, {"params": self.d_net.parameters()}], lr=0.001) if optimizer == "Adam" else optimizer,
            losses=losses,
            loss_weights=loss_weights,
            metrics=metrics,
            decoders=decoders,
            collate_fn=collate_fn
        )

    def summary(self, show_dict=False):

        print(self.name)
        print("Model Parameters:", self.num_params(self.networks))

        print("Representation Network Parameters:", self.num_params(self.r_net))
        if show_dict:
            self.show_dict(self.r_net)

        print("Policy Network Parameters:", self.num_params(self.p_net))
        if show_dict:
            self.show_dict(self.p_net)

        print("Q-Value Network Parameters:", self.num_params(self.q_net))
        if show_dict:
            self.show_dict(self.q_net)

        print("Model Network Parameters:", self.num_params(self.m_net))
        if show_dict:
            self.show_dict(self.m_net)

        print("Encoder Network Parameters:", self.num_params(self.e_net))
        if show_dict:
            self.show_dict(self.e_net)

        print("Decoder Network Parameters:", self.num_params(self.d_net))
        if show_dict:
            self.show_dict(self.d_net)

    def train_step(self, inputs, targets, mixed_precision, grad_scaler, accumulated_steps, acc_step):

        # Train Step
        batch_losses, batch_metrics, _ = super(OnlineImgJDDPG, self).train_step(inputs, targets, mixed_precision, grad_scaler, accumulated_steps, acc_step)

        # Update Target Networks
        for param_target, param_net in zip(self.r_target.parameters(), self.r_net.parameters()):
            param_target.mul_(1 - self.tau)
            param_target.add_(self.tau * param_net.detach())
        for param_target, param_net in zip(self.q_target.parameters(), self.q_net.parameters()):
            param_target.mul_(1 - self.tau)
            param_target.add_(self.tau * param_net.detach())
        for param_target, param_net in zip(self.p_target.parameters(), self.p_net.parameters()):
            param_target.mul_(1 - self.tau)
            param_target.add_(self.tau * param_net.detach())
        for param_target, param_net in zip(self.m_target.parameters(), self.m_net.parameters()):
            param_target.mul_(1 - self.tau)
            param_target.add_(self.tau * param_net.detach())

        # Update Frozen Network
        self.q_frozen.load_state_dict(self.q_net.state_dict())

        # Update Infos
        self.infos["episodes"] = self.episodes
        self.infos["running_rewards"] = round(self.running_rewards, 2)
        self.infos["ep_rewards"] = round(self.ep_rewards, 2)
        self.infos["action_step"] = self.action_step

        return batch_losses, batch_metrics, _

    def forward(self, inputs):

        ###############################################################################
        # Environment Step
        ###############################################################################

        # Get State
        state = self.state.to(self.device)

        # Forward Policy Network
        with torch.no_grad():
            self.p_net.eval()
            action = self.p_net(self.r_net(state.unsqueeze(dim=0)))
            self.p_net.train()

        # Action Info
        self.infos["actions"] = ["{}{:.2f}".format("+" if a >= 0 else "-", abs(a)) for a in action.squeeze(dim=0).tolist()]

        # Update Step
        self.action_step += 1

        # Add Noise
        action += self.noise()

        # Clip Action
        action = action.clip(self.clip_low, self.clip_high)

        # Env Step
        state_next, reward, done = self.env.step(action.squeeze(dim=0))

        # Update ep rewards
        self.ep_rewards += reward

        # Done
        if done:
            self.state = self.env.reset()
            self.episodes += 1
            self.running_rewards = 0.05 * self.ep_rewards + (1 - 0.05) * self.running_rewards
            self.ep_rewards = 0.0
            self.noise.reset()
        else:
            self.state = state_next

        ###############################################################################
        # Model Foward
        ###############################################################################

        # Forward Representation network
        r_state = self.r_net(state.unsqueeze(dim=0))

        # Forward Model
        model_r_state_next, model_reward_pred = self.m_net([r_state, action])

        # Forward Policy network
        model_action_pred = self.p_net(model_r_state_next)

        # Forward Q-Value Network
        model_return_pred = self.q_net([model_r_state_next, model_action_pred])

        # Compute Model Network Targets
        with torch.no_grad():

            # Real Targets
            r_state_next = self.r_target(state_next.unsqueeze(dim=0))
            action_next = self.p_target(r_state_next)
            reward = torch.tensor(reward, dtype=action_next.dtype).reshape(1, 1)

            # Rewards Targets
            self.additional_targets["model_r"] = reward

            # Returns Targets
            self.additional_targets["model_a"] = action_next

            # Returns Targets
            self.additional_targets["model_R"] = reward + self.gamma * self.q_target([r_state_next, action_next])

        ###############################################################################
        # VAE Step
        ###############################################################################

        # Forward Encoder Network
        mean, log_var = self.e_net(r_state)

        # Reparameterize
        vae_r_state_latent = torch.randn(size=mean.size(), device=mean.device) * torch.exp(log_var * 0.5) + mean

        # Forward Decoder Network
        vae_r_state = self.d_net(vae_r_state_latent)

        # VAE Network Targets
        with torch.no_grad():
            self.additional_targets["vae"] = r_state

        ###############################################################################
        # Imaginary Step
        ###############################################################################

        # Generate states
        r_states = self.d_net(torch.randn(size=(self.batch_size,) + mean.shape[1:], device=mean.device)).detach()

        # Forward Policy Network
        pred_actions = self.p_net(r_states)

        # Forward Q-Value Network
        actor_returns_pred = self.q_frozen([r_states, pred_actions])

        # Add Noise to Actions
        actions = pred_actions.detach()#(pred_actions + torch.normal(mean=0, std=self.noise_std, size=actions.size(), device=actions.device)).detach()
        
        # Clip Action
        # actions = actions.clip(self.clip_low, self.clip_high)

        # Forward Q-Value Network
        critic_returns_pred = self.q_net([r_states, actions])

        # Compute Q-Value Network Targets
        with torch.no_grad():
            r_states_next_img, rewards_img = self.m_target([r_states, actions])
            self.additional_targets["critic"] = rewards_img + self.gamma * self.q_target([r_states_next_img, self.p_target(r_states_next_img)])

        return {"actor": - actor_returns_pred, "critic": critic_returns_pred, "model_r": model_reward_pred, "model_a": model_action_pred, "model_R": model_return_pred, "vae": vae_r_state, "KLD": [mean, log_var]}

    def play(self, verbose=0):

        # Reset
        state = self.env.reset().to(self.device)
        total_rewards = 0
        step = 0

        # Episode loop
        while 1:

            # Sample Action
            action = self.p_net(self.r_net(state.unsqueeze(dim=0)))

            # Forward Env
            state, reward, done = self.env.step(action.squeeze(dim=0))
            state = state.to(self.device)
            step += 1
            total_rewards += reward

            # Verbose lvl 1
            if verbose > 0:
                infos = "step {}, action {}, reward {}{:.2f}, done {}, total {:.2f}".format(step, ["{}{:.2f}".format("+" if a >= 0 else "-", abs(a)) for a in action.squeeze(dim=0).tolist()], "+" if reward >=0 else "-", abs(reward), done, total_rewards)
                print(infos)

                # Image State
                if len(state.shape) == 3:
                    plt.title(infos)
                    plt.imshow(state[0].cpu())
                    plt.pause(0.001)
                    plt.close()

            # Verbose lvl 2
            if verbose > 1:
                self.env.env.render()

            # Break
            if done:
                break

        return total_rewards, step

    def eval_step(self, inputs, targets, verbose=0):

        with torch.no_grad():
            score, steps = self.play(verbose=verbose)

        # Update Infos
        self.infos["ep_score"] = round(score, 2)
        self.infos["ep_steps"] = steps

        return {}, {"score": score, "steps": steps}, {}, {}

class DiscreteJDDPG(Model):

    def __init__(self, r_net, p_net, q_net, env, batch_size=64, gamma=0.99, tau=0.001, eps_min=0.1, eps_max=1.0, eps_random=50000, eps_steps=1000000, buffer_size=50000, update_period=1, reward_done=-1, name="Joined Deep Deterministic Policy Gradient Model"):
        super(DiscreteJDDPG, self).__init__(name=name)

        # Representation Networks
        self.r_net = r_net
        self.r_target = type(self.r_net)()
        self.r_target.load_state_dict(self.r_net.state_dict())
        self.set_require_grad(self.r_target, False)

        # Policy Networks
        self.p_net = p_net
        self.p_target = type(self.p_net)()
        self.p_target.load_state_dict(self.p_net.state_dict())
        self.set_require_grad(self.p_target, False)

        # Q-Value Networks
        self.q_net = q_net
        self.q_target = type(self.q_net)()
        self.q_target.load_state_dict(self.q_net.state_dict())
        self.set_require_grad(self.q_target, False)
        self.q_frozen = type(self.q_net)()
        self.q_frozen.load_state_dict(self.q_net.state_dict())
        self.set_require_grad(self.q_frozen, False)

        # Env
        self.env = env
        self.state = self.env.reset()

        # Training Params
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.buffer_size = buffer_size
        self.update_period = update_period
        self.reward_done = reward_done

        # Eps Random Steps
        self.eps_min = eps_min
        self.eps_max = eps_max
        self.eps_inter = eps_max - eps_min
        self.eps = eps_max
        self.eps_random = eps_random
        self.eps_steps = eps_steps

        # Training Infos
        self.episodes = 0
        self.buffer = []
        self.running_rewards = 0.0
        self.ep_rewards = 0.0
        self.action_step = 0

    def compile(
        self, 
        optimizer="Adam",
        losses=[MeanLoss(targets_as_sign=False), MeanSquaredError()],
        loss_weights=[0.1, 1.0],
        metrics=None,
        decoders=None,
        collate_fn=CollateList(inputs_axis=[], targets_axis=[])
    ):
        # Compile Model
        super(DiscreteJDDPG, self).compile(
            optimizer=Adam(params=[{"params": self.r_net.parameters()}, {"params": self.p_net.parameters()}, {"params": self.q_net.parameters()}], lr=0.001) if optimizer == "Adam" else optimizer,
            losses=losses,
            loss_weights=loss_weights,
            metrics=metrics,
            decoders=decoders,
            collate_fn=collate_fn
        )

    def summary(self, show_dict=False):

        print(self.name)
        print("Model Parameters:", self.num_params(self.r_net) + self.num_params(self.p_net) + self.num_params(self.q_net))

        print("Representation Network Parameters:", self.num_params(self.r_net))
        if show_dict:
            self.show_dict(self.r_net)

        print("Policy Network Parameters:", self.num_params(self.p_net))
        if show_dict:
            self.show_dict(self.p_net)

        print("Q-Value Network Parameters:", self.num_params(self.q_net))
        if show_dict:
            self.show_dict(self.q_net)

    def env_step(self):

        # Get State
        state = self.state.to(self.device)

        # Exploration Step
        if self.action_step < self.eps_random or self.eps > torch.rand(1).item():
            action = self.env.sample().unsqueeze(dim=0).to(self.device)

        # Forward Policy Network
        else:
            with torch.no_grad():
                logits = self.p_net(self.r_net(state.unsqueeze(dim=0)))
                action = logits.argmax(dim=-1)

            # Action Info
            self.infos["logits"] = [round(a, 2) for a in logits.squeeze().tolist()]

        # Update Step
        self.action_step += 1

        # Decay probability of taking random action
        self.eps -= self.eps_inter / self.eps_steps
        self.eps = max(self.eps, self.eps_min)

        # Env Step
        state_next, reward, done = self.env.step(action.item())

        # Limit the state and reward history
        if len(self.buffer) >= self.buffer_size:
            del self.buffer[:1]

        # Store Transitions
        self.buffer.append((state, action[0], torch.tensor(reward, device=self.device, dtype=torch.float32), state_next.to(self.device), torch.tensor(done, device=self.device, dtype=torch.float32)))

        # Update ep rewards
        self.ep_rewards += reward

        # Done
        if done:
            self.state = self.env.reset()
            self.episodes += 1
            self.running_rewards = 0.05 * self.ep_rewards + (1 - 0.05) * self.running_rewards
            self.ep_rewards = 0.0
        else:
            self.state = state_next

    def sample_batch(self):

        # Sample Batch from Buffer
        samples = [self.buffer[i] for i in torch.randint(low=0, high=len(self.buffer), size=(self.batch_size,))]
        states = torch.stack([sample[0] for sample in samples])
        actions = torch.stack([sample[1] for sample in samples])
        rewards = torch.stack([sample[2] for sample in samples])
        states_next = torch.stack([sample[3] for sample in samples])
        dones = torch.stack([sample[4] for sample in samples])

        return [states, actions, rewards, states_next, dones]

    def train_step(self, inputs, targets, mixed_precision, grad_scaler, accumulated_steps, acc_step):

        # Environment Step
        env_step = 0
        while env_step < self.update_period or len(self.buffer) < self.batch_size:
          self.env_step()
          env_step += 1

        # Sample Inputs / Targets Batch
        inputs = self.sample_batch()

        # Update Infos
        self.infos["episodes"] = self.episodes
        self.infos["running_rewards"] = round(self.running_rewards, 2)
        self.infos["ep_rewards"] = round(self.ep_rewards, 2)
        self.infos["eps"] = round(self.eps, 2)
        self.infos["action_step"] = self.action_step

        # Train Step
        batch_losses, batch_metrics, _ = super(DiscreteJDDPG, self).train_step(inputs, targets, mixed_precision, grad_scaler, accumulated_steps, acc_step)

        # Update Target Networks
        for param_target, param_net in zip(self.r_target.parameters(), self.r_net.parameters()):
            param_target.mul_(1 - self.tau)
            param_target.add_(self.tau * param_net.detach())
        for param_target, param_net in zip(self.q_target.parameters(), self.q_net.parameters()):
            param_target.mul_(1 - self.tau)
            param_target.add_(self.tau * param_net.detach())
        for param_target, param_net in zip(self.p_target.parameters(), self.p_net.parameters()):
            param_target.mul_(1 - self.tau)
            param_target.add_(self.tau * param_net.detach())

        # Update Frozen Network
        self.q_frozen.load_state_dict(self.q_net.state_dict())

        return batch_losses, batch_metrics, _

    def forward(self, inputs):

        # Unpack Inputs
        states, actions, rewards, states_next, dones = inputs

        #print(states, actions, rewards, states_next)

        # Forward Representation Network
        r_states = self.r_net(states)

        # Forward Policy Network
        pred_actions_logits = self.p_net(r_states)

        # Forward Q-Value Network: Policy Loss
        actor_pred_returns = self.q_frozen(r_states, pred_actions_logits, from_logits=True)

        # Forward Q-Value Network: Q-Value Loss
        critic_pred_returns = self.q_net(r_states, actions, from_logits=False)

        # Compute Q-Value Network Targets
        with torch.no_grad():
            r_states_next = self.r_target(states_next)
            self.additional_targets["critic"] = (rewards.unsqueeze(-1) + self.gamma * self.q_target(r_states_next, self.p_target(r_states_next).argmax(dim=-1), from_logits=False)) * (1 - dones.unsqueeze(-1)) + self.reward_done * dones.unsqueeze(-1)

        # Action Diversity
        self.infos["diversity"] = round(100 * pred_actions_logits.argmax(dim=-1).unique().numel() / pred_actions_logits.size(-1), 2)

        return {"actor": - actor_pred_returns, "critic": critic_pred_returns}

    def play(self, verbose=False):

        # Reset
        env = copy.deepcopy(self.env)
        state = env.reset().to(self.device)
        total_rewards = 0
        step = 0

        # Episode loop
        while 1:

            # Sample Action
            action = self.p_net(self.r_net(state.unsqueeze(dim=0))).argmax(dim=-1)

            # Forward Env
            state, reward, done = env.step(action.item())
            state = state.to(self.device)
            step += 1
            total_rewards += reward

            # Verbose
            if verbose:
                infos = "step {}, action {}, reward {:.2f}, done {}, total {:.2f}".format(step, action, reward, done, total_rewards)
                print(infos)

                # Image State
                if len(state.shape) == 3:
                    plt.title(infos)
                    plt.imshow(state[0].cpu())
                    plt.pause(0.001)
                    plt.close()

            # Break
            if done:
                break

        return total_rewards, step

    def eval_step(self, inputs, targets, verbose=False):

        with torch.no_grad():
            score, steps = self.play(verbose=verbose)

        # Update Infos
        self.infos["ep_score"] = round(score, 2)
        self.infos["ep_steps"] = steps

        return {}, {"score": score, "steps": steps}, {}, {}