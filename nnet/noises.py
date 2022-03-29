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
import torch.nn as nn

###############################################################################
# Noise Process
###############################################################################

class GaussianNoise(nn.Module):

    def __init__(self, mean:float=0.0, std:float=1.0, dim_noise:tuple=(100,)):
        super(GaussianNoise, self).__init__()

        # Mean / Std
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32))

        # Dim Noise
        self.dim_noise = dim_noise

    def forward(self, batch_size=1):

        return torch.normal(mean=self.mean, std=self.std, size=(batch_size,) + self.dim_noise, device=self.mean.device)

class OrnsteinUhlenbeckProcess(nn.Module):

    """ 
    Ornstein-Uhlenbeck process noise for DDPG
    https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.

    """

    def __init__(self, num_actions, mean=0, std=0.2, theta=0.15, dt=0.05):
        super(OrnsteinUhlenbeckProcess, self).__init__()

        # Params
        self.num_actions = num_actions
        self.mean = mean
        self.std = std
        self.theta = theta
        self.dt = dt

        # Regiser Noise Buffer
        self.register_buffer("noise", torch.zeros(self.num_actions))

    def __call__(self):

        self.noise.add_(self.theta * (self.mean - self.noise) * self.dt + self.std * self.dt ** 0.5 * torch.randn(self.num_actions, device=self.noise.device))

        return self.noise

    def reset(self):

        self.noise.fill_(0.0)