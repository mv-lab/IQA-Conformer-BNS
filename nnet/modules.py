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
import torch.nn.functional as F

# Module
from nnet.module import Module

# Attentions
from nnet.attentions import (
    att_dict,
    # Abs Attentions
    MultiHeadAttention,
    AxialMultiHeadAttention,
    GroupedMultiHeadAttention,
    LocalMultiHeadAttention,
    StridedMultiHeadAttention,
    StridedLocalMultiHeadAttention,
    MultiHeadLinearAttention,
    # Rel Attentions
    RelPosMultiHeadSelfAttention,
    GroupedRelPosMultiHeadSelfAttention,
    LocalRelPosMultiHeadSelfAttention,
    StridedRelPosMultiHeadSelfAttention,
    StridedLocalRelPosMultiHeadSelfAttention
)

# Layers
from nnet.layers import (
    Embedding,
    layer_dict,
    Linear,
    Conv1d,
    Conv2d,
    Conv3d,
    ConvTranspose1d,
    ConvTranspose2d,
    ConvTranspose3d,
    Transpose,
    Permute,
    Reshape,
    GlobalAvgPool1d,
    GlobalAvgPool2d,
    GlobalAvgPool3d,
    Dropout,
    SwitchLinear
)

# Activations
from nnet.activations import (
    act_dict,
    Swish
)

# Normalization Layers
from nnet.normalizations import (
    norm_dict,
    BatchNorm1d,
    BatchNorm2d,
    BatchNorm3d,
    LayerNorm
)

# Initializations
from nnet.initializations import (
    init_dict
)

# Schedulers
from nnet.schedulers import (
    Scheduler,
    ConstantScheduler,
    LinearDecayScheduler
)

# Other
from typing import Union

###############################################################################
# Transformer Modules
###############################################################################

class FeedForwardModule(nn.Module):

    """Transformer Feed Forward Module

    Args:
        dim_model: model feature dimension
        dim_ffn: expanded feature dimension
        Pdrop: dropout probability
        act: inner activation function
        inner_dropout: whether to apply dropout after the inner activation function

    Input: (batch size, length, dim_model)
    Output: (batch size, length, dim_model)
    
    """

    def __init__(self, dim_model, dim_ffn, drop_rate, act_fun, inner_dropout, prenorm=True):
        super(FeedForwardModule, self).__init__()

        # Layers
        self.layers = nn.Sequential(
            nn.LayerNorm(dim_model) if prenorm else nn.Identity(),
            Linear(dim_model, dim_ffn),
            act_dict[act_fun](),
            nn.Dropout(p=drop_rate) if inner_dropout else nn.Identity(),
            Linear(dim_ffn, dim_model),
            nn.Dropout(p=drop_rate)
        )

    def forward(self, samples, output_dict=False, **kargs):

        # Layers
        samples = self.layers(samples)

        return {"samples": samples} if output_dict else samples

class AttentionModule(nn.Module):

    """ Attention Module

    Args:
        dim_model: model feature dimension
        att_params: attention params
        drop_rate: residual dropout probability
        max_pos_encoding: maximum position
        causal: True for causal attention with masked future context

    """

    def __init__(self, dim_model, att_params, drop_rate, max_pos_encoding, causal):
        super(AttentionModule, self).__init__()

        # Pre Norm
        self.norm = nn.LayerNorm(dim_model)

        # Attention
        self.attention = att_dict[att_params["class"]](dim_model=dim_model, causal=causal, max_pos_encoding=max_pos_encoding, **att_params)
            
        # Dropout
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x, x_cross=None, mask=None, hidden=None):

        # Pre Norm
        x = self.norm(x)

        # Self-Attention
        x, att_maps, hidden = self.attention(Q=x, K=x_cross if x_cross != None else x, V=x_cross if x_cross != None else x, mask=mask, hidden=hidden)

        # Dropout
        x = self.dropout(x)

        return x, att_maps, hidden

class MultiHeadSelfAttentionModule(nn.Module):

    """Multi-Head Self-Attention Module

    Args:
        dim_model: model feature dimension
        num_heads: number of attention heads
        Pdrop: residual dropout probability
        max_pos_encoding: maximum position
        relative_pos_enc: whether to use relative postion embedding
        causal: True for causal attention with masked future context
        group_size: Attention group size
        kernel_size: Attention kernel size
        stride: Query stride
        linear_att: whether to use multi-head linear self-attention
        axis: attention axis for axial attention

    """

    def __init__(self, dim_model, num_heads, Pdrop, max_pos_encoding, relative_pos_enc, causal, group_size, kernel_size, stride, linear_att, axis):
        super(MultiHeadSelfAttentionModule, self).__init__()

        # Assert
        assert not (group_size > 1 and kernel_size != None), "Local grouped attention not implemented"
        assert not (group_size > 1 and stride > 1), "Strided grouped attention not implemented"
        assert not (linear_att and relative_pos_enc), "Linear attention requires absolute positional encodings"

        # Pre Norm
        self.norm = nn.LayerNorm(dim_model)

        # Axial Multi-Head Attention
        if axis != None:
            self.mhsa = AxialMultiHeadAttention(dim_model, num_heads)

        # Multi-Head Linear Attention
        elif linear_att:
            self.mhsa = MultiHeadLinearAttention(dim_model, num_heads)

        # Grouped Multi-Head Self-Attention
        elif group_size > 1:
            if relative_pos_enc:
                self.mhsa = GroupedRelPosMultiHeadSelfAttention(dim_model, num_heads, causal, max_pos_encoding, group_size)
            else:
                self.mhsa = GroupedMultiHeadAttention(dim_model, num_heads, group_size)
        
        # Local Multi-Head Self-Attention
        elif kernel_size is not None and stride == 1:
            if relative_pos_enc:
                self.mhsa = LocalRelPosMultiHeadSelfAttention(dim_model, num_heads, causal, kernel_size)
            else:
                self.mhsa = LocalMultiHeadAttention(dim_model, num_heads, kernel_size)

        # Strided Multi-Head Self-Attention
        elif kernel_size is None and stride > 1:
            if relative_pos_enc:
                self.mhsa = StridedRelPosMultiHeadSelfAttention(dim_model, num_heads, causal, max_pos_encoding, stride)
            else:
                self.mhsa = StridedMultiHeadAttention(dim_model, num_heads, stride)

        # Strided Local Multi-Head Self-Attention
        elif stride > 1 and kernel_size is not None:
            if relative_pos_enc:
                self.mhsa = StridedLocalRelPosMultiHeadSelfAttention(dim_model, num_heads, causal, kernel_size, stride)
            else:
                self.mhsa = StridedLocalMultiHeadAttention(dim_model, num_heads, kernel_size, stride)

        # Multi-Head Self-Attention
        else:
            if relative_pos_enc:
                self.mhsa = RelPosMultiHeadSelfAttention(dim_model, num_heads, causal, max_pos_encoding)
            else:
                self.mhsa = MultiHeadAttention(dim_model, num_heads)
            
        # Dropout
        self.dropout = nn.Dropout(Pdrop)

        # Module Params
        self.rel_pos_enc = relative_pos_enc
        self.linear_att = linear_att

    def forward(self, samples, mask=None, hidden=None, output_dict=False, **args):

        # Pre Norm
        samples = self.norm(samples)

        # Multi-Head Self-Attention
        if self.linear_att:
            samples, attention = self.mhsa(samples, samples, samples)
        elif self.rel_pos_enc:
            samples, attention, hidden = self.mhsa(samples, samples, samples, mask, hidden)
        else:
            samples, attention = self.mhsa(samples, samples, samples, mask)

        # Dropout
        samples = self.dropout(samples)

        return {"samples": samples, "attention": attention, "hidden": hidden} if output_dict else (samples, attention, hidden)

class MultiHeadCrossAttentionModule(nn.Module):

    """Multi-Head Cross-Attention Module

    Args:
        dim_model: model feature dimension
        num_heads: number of attention heads
        Pdrop: residual dropout probability

    """

    def __init__(self, dim_model, num_heads, Pdrop):
        super(MultiHeadCrossAttentionModule, self).__init__()

        # Pre Norm
        self.norm = nn.LayerNorm(dim_model)

        # Multi-Head Cros-Attention
        self.mhca = MultiHeadAttention(dim_model, num_heads)
            
        # Dropout
        self.dropout = nn.Dropout(Pdrop)

    def forward(self, x, x_enc, mask_enc=None):

        # Pre Norm
        x = self.norm(x)

        # Multi-Head Cross-Attention
        x, attention = self.mhca(Q=x, K=x_enc, V=x_enc, mask=mask_enc)

        # Dropout
        x = self.dropout(x)

        return x, attention

class ConvolutionModule(nn.Module):

    """Conformer Convolution Module

    Args:
        dim_model: input feature dimension
        dim_expand: output feature dimension
        kernel_size: depthwise convolution kernel size
        drop_rate: residual dropout probability
        stride: depthwise convolution stride
        padding: "valid", "same" or "causal"
        dim: number of spatiotemporal input dimensions
        channels_last: ordering of the dimensions in the inputs

    References: 
        https://arxiv.org/abs/2005.08100
    
    """

    def __init__(self, dim_model, dim_expand, drop_rate, stride, act_fun="Swish", conv_params={"class": "Conv2d", "params":{"padding":"same", "kernel_size": 3}}, channels_last=False):
        super(ConvolutionModule, self).__init__()

        # Layers
        pointwise_conv = layer_dict[conv_params["class"].replace("Transpose", "")]
        depthwise_conv = layer_dict[conv_params["class"]]
        norm = norm_dict[conv_params["class"].replace("Transpose", "").replace("Conv", "BatchNorm")]

        # Layers
        self.layers = nn.Sequential(
            LayerNorm(dim_model, channels_last=channels_last),
            pointwise_conv(dim_model, 2 * dim_expand, kernel_size=1, channels_last=channels_last),
            nn.GLU(dim=-1 if channels_last else 1),
            depthwise_conv(dim_expand, dim_expand, stride=stride, groups=dim_expand, channels_last=channels_last, **conv_params["params"]),
            norm(dim_expand, channels_last=channels_last),
            act_dict[act_fun](),
            pointwise_conv(dim_expand, dim_expand, kernel_size=1, channels_last=channels_last),
            nn.Dropout(p=drop_rate)
        )

    def forward(self, x):

        return self.layers(x)

class PatchEmbedding(nn.Module):

    def __init__(self, input_channels, patch_size, embedding_dim):
        super(PatchEmbedding, self).__init__()

        # Patch Embedding
        self.embedding = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=embedding_dim, kernel_size=patch_size, stride=patch_size),
            nn.Flatten(start_dim=2),
            Transpose(1, 2)
        )

    def forward(self, x):

        # Patch Embedding
        return self.embedding(x)
