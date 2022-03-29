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

from nnet.module import Module

from nnet.schedulers import (
    Scheduler,
    ConstantScheduler,
    LinearDecayScheduler
)

from nnet.layers import (
    Linear,
    Embedding
)

from nnet.initializations import (
    init_dict
)

###############################################################################
# Vector Quantization
###############################################################################

class VectorQuantizer(Module):

    """ Vector Quantizer with Exp Moving Average
    
        Reference: "Neural Discrete Representation Learning" by Aaron van den Oord et al.
        https://arxiv.org/abs/1711.00937
        https://github.com/deepmind/sonnet/blob/v1/sonnet/python/modules/nets/vqvae.py
    
    """

    def __init__(self, num_embeddings, embedding_dim, init="lecun_uniform", beta=0.25):
        super(VectorQuantizer, self).__init__()

        # Embeddings (D, K)
        self.embeddings = nn.Parameter(torch.Tensor(embedding_dim, num_embeddings))

        # Init
        init_dict[init](self.embeddings.transpose(0, 1))

        # num_embeddings
        self.num_embeddings = num_embeddings
        
        # embedding_dim
        self.embedding_dim = embedding_dim

        # Commitment Loss Coef
        self.beta = beta

    def forward(self, x):

        # Encode
        indices = self.encode(x)

        # Diversity
        self.add_info("diversity", round(100 * indices.unique().numel() / self.num_embeddings, 2))

        # Decode
        quantized = self.decode(indices)

        if self.training:

            # Add Losses
            self.add_loss("codebook", F.mse_loss(quantized, x.detach()))
            self.add_loss("commit", self.beta * F.mse_loss(x, quantized.detach()))

            # Gradient Identity
            quantized = x + (quantized - x).detach()

        return quantized

    def encode(self, x):

        # Flatten (..., D) -> (-1, D)
        x_flatten = x.flatten(start_dim=0, end_dim=-2)

        # Compute Distances
        dis = x_flatten.square().sum(dim=-1, keepdim=True) + self.embeddings.square().sum(dim=0, keepdim=True) - 2 * x_flatten.matmul(self.embeddings)

        # Compute Indices (-1)
        indices = dis.argmin(dim=-1)

        # Unflatten (-1) -> (...)
        indices = indices.reshape(x.shape[:-1])

        return indices

    def decode(self, indices):

        # Quantize (...) -> (..., D)
        quantized = F.embedding(indices, self.embeddings.transpose(0, 1))

        return quantized

class VectorQuantizerEMA(VectorQuantizer):

    """ Vector Quantizer with Exp Moving Average
    
        Reference: "Neural Discrete Representation Learning" by Aaron van den Oord et al.
        https://arxiv.org/abs/1711.00937
        https://github.com/deepmind/sonnet/blob/v1/sonnet/python/modules/nets/vqvae.py
    
    """

    def __init__(self, num_embeddings, embedding_dim, init="lecun_uniform", beta=0.25, gamma=0.99, eps=1e-5):
        super(VectorQuantizerEMA, self).__init__(num_embeddings, embedding_dim, init, beta)

        # Params
        self.gamma = gamma
        self.eps = eps

        # Buffers
        self.register_buffer("N", torch.Tensor(num_embeddings)) # (num_embeddings,)
        self.register_buffer("m", torch.Tensor(embedding_dim, num_embeddings)) # (embedding_dim, num_embeddings)

        #Init
        nn.init.zeros_(self.N)
        self.m.copy_(self.embeddings)
        
    def forward(self, x):

        # Encode
        indices = self.encode(x)

        # Diversity
        self.add_info("diversity", round(100 * indices.unique().numel() / self.num_embeddings, 2))

        # Decode
        quantized = self.decode(indices).detach()

        if self.training:

            # Add Losses
            self.add_loss("commit", self.beta * F.mse_loss(x, quantized.detach()))

            # Gradient Identity
            quantized = x + (quantized - x).detach()

            # Reduce indices among devices (B, ...) -> (B * n_devices, ...)
            if torch.distributed.is_initialized():
                indices_list = [torch.zeros_like(indices) for _ in range(torch.distributed.get_world_size())]
                torch.distributed.all_gather(indices_list, indices)
                indices = torch.cat(indices_list, dim=0)

            # Update Codebook
            with torch.no_grad():

                # Flatten and One Hot Indices (...) -> (N, K)
                indices_one_hot = F.one_hot(indices.reshape(-1).type(torch.long), num_classes=self.num_embeddings).type(self.m.dtype)

                # Update Buffers
                self.N = self.gamma * self.N + (1 - self.gamma) * indices_one_hot.sum(0)
                self.m = self.gamma * self.m + (1 - self.gamma) * x.flatten(start_dim=0, end_dim=-2).transpose(0,1).matmul(indices_one_hot)

                # Scale Count
                N_sum = self.N.sum()
                N_scaled = (self.N + self.eps) / (N_sum + self.num_embeddings * self.eps) * N_sum

                # Update Embeddings
                self.embeddings.copy_(self.m / N_scaled)

        return quantized

class GumbelSoftmaxQuantizer(Module):

    """ Gumbel Softmax Vector Quantizer
    
        Reference: "CATEGORICAL REPARAMETERIZATION WITH GUMBEL-SOFTMAX" by Jang et al. 2016
        https://arxiv.org/abs/1611.01144
    
    """

    def __init__(self, num_embeddings, embedding_dim, init="lecun_uniform", tau=LinearDecayScheduler(value_start=2, value_end=0.5, decay_steps=70000), gumbel_eps=1e-10):
        super(GumbelSoftmaxQuantizer, self).__init__()

        # Embeddings (D, K)
        self.embeddings = nn.Parameter(torch.Tensor(embedding_dim, num_embeddings))

        # Init
        init_dict[init](self.embeddings.transpose(0, 1))

        # num_embeddings
        self.num_embeddings = num_embeddings
        
        # embedding_dim
        self.embedding_dim = embedding_dim

        # Network
        self.net = nn.Sequential(
            Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            Linear(embedding_dim, num_embeddings)
        )

        # Tau
        if isinstance(tau, Scheduler):
            self.tau = tau
        else:
            self.tau = ConstantScheduler(lr_value=tau)

        # Eps
        self.gumbel_eps = gumbel_eps

    def forward(self, x, return_indices=False):

        if self.training:
            indices, logits = self.encode(x, return_logits=True)
            quantized = self.decode(indices, logits)
        else:
            indices = self.encode(x)
            quantized = self.decode(indices)

        # Diversity
        self.add_info("diversity", round(100 * indices.unique().numel() / self.num_embeddings, 2))

        return (quantized, indices) if return_indices else quantized

    def encode(self, x, return_logits=False):

        # Compute Logits (..., K)
        logits = self.net(x)

        # Compute Indices (...)
        indices = logits.argmax(dim=-1)

        return (indices, logits) if return_logits else indices

    def decode(self, indices, logits=None):

        if logits != None:

            # Update Tau
            tau = self.tau.step()
            self.add_info("tau", round(tau.item(), 2))

            # Hardmax (..., K)
            y_hard = F.one_hot(indices, num_classes=self.num_embeddings)

            # Gumbel Noise
            gumbel_noise = - torch.log( - torch.log(logits.new_empty(size=logits.size()).uniform_(0, 1) + self.gumbel_eps) + self.gumbel_eps)

            # Softmax
            y_soft = ((logits - gumbel_noise) / tau).softmax(dim=-1)

            # Reparameterization Trick
            y = (y_hard - y_soft).detach() + y_soft
            
            # Quantize (..., K) -> (..., D)
            quantized = y.matmul(self.embeddings.transpose(0, 1))

        else:

            # Quantize (...) -> (..., D)
            quantized = F.embedding(indices, self.embeddings.transpose(0, 1))

        return quantized

class GumbelSoftmaxEmbedding(Module):

    def __init__(self, num_embeddings, embedding_dim, tau=1):
        super(GumbelSoftmaxEmbedding, self).__init__()

        # Embeddings (D, K)
        self.embeddings = Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)

        # Tau
        if isinstance(tau, Scheduler):
            self.tau = tau
        else:
            self.tau = ConstantScheduler(lr_value=tau)

    def forward(self, x, from_logits=False):

        if from_logits:

            # Update Tau
            tau = float(self.tau.step())

            # Infos
            self.add_info("tau", round(tau, 2))

            # Gumbel Softmax
            y = F.gumbel_softmax(x, tau=tau, dim=-1)
            
            # Embedding (..., K) -> (..., D)
            x_emb = y.matmul(self.embeddings.weight)

        else:

            # Embedding (...) -> (..., D)
            x_emb = self.embeddings(x)

        return x_emb