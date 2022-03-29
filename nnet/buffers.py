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

import torch
import torch.nn as nn
import torch.nn.functional as F

###############################################################################
# Experience Replay Buffers
###############################################################################

class ReplayBuffer(nn.Module):

    def __init__(self, size: int, state_size: tuple, action_size: tuple):
        super(ReplayBuffer, self).__init__()

        # Params
        self.register_buffer("size", torch.tensor(size))
        self.register_buffer("num_elt", torch.tensor(0))
        self.register_buffer("index", torch.tensor(0))

        # Init Buffers
        self.register_buffer("states", torch.zeros((self.size,) + state_size))
        self.register_buffer("actions", torch.zeros((self.size,) + action_size))
        self.register_buffer("rewards", torch.zeros(self.size))
        self.register_buffer("states_next", torch.zeros((self.size,) + state_size))
        self.register_buffer("dones", torch.zeros(self.size))

    def append(self, state, action, reward, state_next, done):

        # Append Element
        self.states[self.index] = state
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.states_next[self.index] = state_next
        self.dones[self.index] = done

        # Update Index
        self.index = torch.tensor(0) if self.index == self.size - 1 else self.index + 1

        # Update Number of Elements
        self.num_elt = min(self.num_elt + 1, self.size)

    def sample(self, batch_size=1):

        # Sample Indices (B,)
        indices = torch.randint(low=0, high=self.num_elt, size=(batch_size,))

        # Select Elements
        states = self.states[indices]
        actions = self.actions[indices]
        rewards = self.rewards[indices]
        states_next = self.states_next[indices]
        dones = self.dones[indices]

        #print("states", states)
        #print("actions", actions)
        #print("rewards", rewards)
        #print("states_next", states_next)
        #print("dones", dones)
        #print()
        #print(self.rewards, self.num_elt)
        #exit()

        return states, actions, rewards, states_next, dones

class PrioritizedReplayBuffer(ReplayBuffer):

    """ Prioritized Replay Buffer

        Reference: "PRIORITIZED EXPERIENCE REPLAY" by Schaul et al. 2015
        https://arxiv.org/abs/1511.05952
    
    """

    def __init__(self, size, state_size, action_size, alpha=0.6, beta=0.4, sigma_eps=1e-10, eps=1e-10):
        super(PrioritizedReplayBuffer, self).__init__(size, state_size, action_size)

        # Sigmas
        self.register_buffer("sigmas", torch.zeros(size, dtype=torch.float32))

        # Params
        self.alpha = alpha
        self.beta = beta # To convrege up to 1
        self.sigma_eps = sigma_eps
        self.eps = eps

    def append(self, state, action, reward, state_next, done):

        # Init Sigma with Maximum Priority
        self.sigmas[self.index] = max(self.sigma_eps, self.sigmas.max())

        # Append
        super(PrioritizedReplayBuffer, self).append(state, action, reward, state_next, done)

    def update_sigmas(self, sigmas, indices):

        # Update Sigmas
        self.sigmas[indices] = sigmas + self.eps

    def sample(self, batch_size=1):

        # Compute Sigmas Probs
        sigmas_probs = (self.sigmas[:self.num_elt] ** self.alpha) / (self.sigmas[:self.num_elt].sum() ** self.alpha + self.eps)

        # Sample (B,)
        indices = sigmas_probs.multinomial(num_samples=batch_size, replacement=True)

        # Compute Weights: Lower High Prob loss / Increase Low Prob loss
        weights = (self.num_elt * sigmas_probs) ** (- self.beta)
        weights /= weights.max()

        # Select Elements
        states = self.states[indices]
        actions = self.actions[indices]
        rewards = self.rewards[indices]
        states_next = self.states_next[indices]
        dones = self.dones[indices]
        weights = weights[indices]

        #print(weights)
        #print(sigmas_probs[indices])

        return states, actions, rewards, states_next, dones, indices, weights