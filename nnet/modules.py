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
# MLP Modules
###############################################################################

class MultiLayerPerceptron(nn.Module):

    def __init__(self, dim_input: int, dim_layers: list, act_fun: Union[str, nn.Module], norm: dict, drop_rate: int):
        super(MultiLayerPerceptron, self).__init__()

        # Act fun
        if isinstance(act_fun, dict):
            act_fun_params = act_fun["params"]
            act_fun = act_dict[act_fun["class"]]
        else:
            act_fun_params = {}
            act_fun = act_dict[act_fun]

        # Norm
        if isinstance(norm, dict):
            norm_params = norm["params"]
            norm = norm_dict[norm["class"]]
        else:
            norm_params = {}
            norm = norm_dict[norm]
            
        # MLP Layers
        self.layers = nn.ModuleList([nn.Sequential(
            Linear(dim_input if layer_id == 0 else dim_layers[layer_id - 1], dim_layers[layer_id]),
            norm(dim_layers[layer_id], **norm_params),
            act_fun(**act_fun_params),
            nn.Dropout(drop_rate)
        ) for layer_id in range(len(dim_layers))])

    def forward(self, x):

        # Layers
        for layer in self.layers:
            x = layer(x)

        return x

###############################################################################
# CNN Modules
###############################################################################

class ConvNeuralNetwork(nn.Module):

    def __init__(self, dim_input, dim_layers, kernel_size, strides, act_fun, norm, drop_rate, padding="same", dim=2, channels_last=False, residual=False):
        super(ConvNeuralNetwork, self).__init__()

        conv = {
            1: Conv1d,
            2: Conv2d,
            3: Conv3d
        }

        # Act fun
        if isinstance(act_fun, dict):
            act_fun_params = act_fun["params"]
            act_fun = act_dict[act_fun["class"]]
        else:
            act_fun_params = {}
            act_fun = act_dict[act_fun]

        # Norm
        if isinstance(norm, dict):
            norm_params = norm["params"]
            norm = norm_dict[norm["class"]]
        else:
            norm_params = {}
            norm = norm_dict[norm]

        # Strides
        self.strides = strides

        # Residual
        self.residual = residual

        # CNN Layers
        self.strides = strides
        self.layers = nn.ModuleList([nn.Sequential(
            conv[dim](dim_input if layer_id == 0 else dim_layers[layer_id - 1], dim_layers[layer_id], kernel_size, stride=self.strides[layer_id], padding=padding, channels_last=channels_last), 
            norm(dim_layers[layer_id], **norm_params, channels_last=channels_last),
            act_fun(**act_fun_params),
            nn.Dropout(drop_rate)
        ) for layer_id in range(len(dim_layers))])

    def forward(self, x, x_len=None):

        # Layers
        for layer in self.layers:

            # Forward
            if self.residual:
                x = x + layer(x)
            else:
                x = layer(x)

            # Update Sequence Lengths
            if x_len is not None:
                x_len = torch.div(x_len - 1, 2, rounding_mode='floor') + 1 # to generalize

        return x if x_len==None else (x, x_len)

class ConvTransposeNeuralNetwork(nn.Module):

    def __init__(self, dim_input, dim_layers, kernel_size, strides, act_fun, norm, drop_rate, padding, output_padding, dim=2, channels_last=False):
        super(ConvTransposeNeuralNetwork, self).__init__()

        conv = {
            1: nn.ConvTranspose1d,
            2: nn.ConvTranspose2d,
            3: nn.ConvTranspose3d
        }

        # Act fun
        if isinstance(act_fun, dict):
            act_fun_params = act_fun["params"]
            act_fun = act_dict[act_fun["class"]]
        else:
            act_fun_params = {}
            act_fun = act_dict[act_fun]

        # Norm
        if isinstance(norm, dict):
            norm_params = norm["params"]
            norm = norm_dict[norm["class"]]
        else:
            norm_params = {}
            norm = norm_dict[norm]

        # Strides
        self.strides = strides

        # CNN Layers
        self.strides = strides
        self.layers = nn.ModuleList([nn.Sequential(
            conv[dim](dim_input if layer_id == 0 else dim_layers[layer_id - 1], dim_layers[layer_id], kernel_size, stride=self.strides[layer_id], padding=padding, output_padding=output_padding[layer_id]), 
            norm(dim_layers[layer_id], **norm_params, channels_last=channels_last),
            act_fun(**act_fun_params),
            nn.Dropout(drop_rate)
        ) for layer_id in range(len(dim_layers))])

    def forward(self, x, x_len=None):

        # Layers
        for layer in self.layers:

            x = layer(x)

        return x if x_len==None else (x, x_len)

class Conv2dPoolSubsampling(nn.Module):

    """Conv2d with Max Pooling Subsampling Block

    Args:
        num_layers: number of strided convolution layers
        filters: list of convolution layers filters
        kernel_size: convolution kernel size
        norm: normalization
        act: activation function

    Shape:
        Input: (batch_size, in_dim, in_length)
        Output: (batch_size, out_dim, out_length)
    
    """

    def __init__(self, num_layers, filters, kernel_size, norm, act_fun):
        super(Conv2dPoolSubsampling, self).__init__()

        # Assert
        assert norm in ["batch", "layer", "none"]

        # Layers
        self.layers = nn.ModuleList([nn.Sequential(
            nn.Conv2d(1 if layer_id == 0 else filters[layer_id - 1], filters[layer_id], kernel_size, padding=(kernel_size - 1) // 2),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(filters[layer_id]) if norm == "batch" else nn.LayerNorm(filters[layer_id]) if norm == "layer" else nn.Identity(),
            act_dict[act_fun]()
        ) for layer_id in range(num_layers)])

    def forward(self, x, x_len):

        # (B, D, T) -> (B, 1, D, T)
        x = x.unsqueeze(dim=1)

        # Layers
        for layer in self.layers:
            x = layer(x)

            # Update Sequence Lengths
            if x_len is not None:
                x_len = (x_len - 1) // 2 + 1

        # (B, C, D // S, T // S) -> (B,  C * D // S, T // S)
        batch_size, channels, subsampled_dim, subsampled_length = x.size()
        x = x.reshape(batch_size, channels * subsampled_dim, subsampled_length)

        return x, x_len

class VGGSubsampling(nn.Module):

    """VGG style Subsampling Block

    Args:
        num_layers: number of strided convolution layers
        filters: list of convolution layers filters
        kernel_size: convolution kernel size
        norm: normalization
        act: activation function

    Shape:
        Input: (batch_size, in_dim, in_length)
        Output: (batch_size, out_dim, out_length)
    
    """

    def __init__(self, num_layers, filters, kernel_size, norm, act_fun):
        super(VGGSubsampling, self).__init__()

        # Assert
        assert norm in ["batch", "layer", "none"]

        self.layers = nn.ModuleList([nn.Sequential(
            # Conv 1
            nn.Conv2d(1 if layer_id == 0 else filters[layer_id - 1], filters[layer_id], kernel_size, padding=(kernel_size - 1) // 2),
            nn.BatchNorm2d(filters[layer_id]) if norm == "batch" else nn.LayerNorm(filters[layer_id]) if norm == "layer" else nn.Identity(),
            act_dict[act_fun](),
            # Conv 2
            nn.Conv2d(filters[layer_id], filters[layer_id], kernel_size, padding=(kernel_size - 1) // 2),
            nn.BatchNorm2d(filters[layer_id]) if norm == "batch" else nn.LayerNorm(filters[layer_id]) if norm == "layer" else nn.Identity(),
            act_dict[act_fun](),
            # Pooling
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)) 
        ) for layer_id in range(num_layers)])

    def forward(self, x, x_len):

        # (B, D, T) -> (B, 1, D, T)
        x = x.unsqueeze(dim=1)

        # Stages
        for layer in self.layers:
            x = layer(x)

            # Update Sequence Lengths
            if x_len is not None:
                x_len = x_len // 2

        # (B, C, D // S, T // S) -> (B,  C * D // S, T // S)
        batch_size, channels, subsampled_dim, subsampled_length = x.size()
        x = x.reshape(batch_size, channels * subsampled_dim, subsampled_length)
        
        return x, x_len

###############################################################################
# Residual CNN Modules
###############################################################################
            
class InvResBlock(nn.Module):

    """MobileNetV2 Inverted Residual Block

    References: 
        https://arxiv.org/abs//1801.04381
    
    """

    def __init__(self, in_features, out_features, expand_ratio, kernel_size, stride, act_fun, dim=2, channels_last=False):
        super(InvResBlock, self).__init__()

        conv = {
            1: Conv1d,
            2: Conv2d,
            3: Conv3d
        }

        norm = {
            1: BatchNorm1d,
            2: BatchNorm2d,
            3: BatchNorm3d
        }

        # Layers
        self.layers = nn.Sequential(
            conv[dim](in_channels=in_features, out_channels=in_features * expand_ratio, kernel_size=1, channels_last=channels_last),
            norm[dim](num_features=in_features * expand_ratio, channels_last=channels_last),
            act_dict[act_fun](),
            conv[dim](in_channels=in_features * expand_ratio, out_channels=in_features * expand_ratio, kernel_size=kernel_size, stride=stride, groups=in_features * expand_ratio, channels_last=channels_last),
            norm[dim](num_features=in_features * expand_ratio, channels_last=channels_last),
            act_dict[act_fun](),
            conv[dim](in_channels=in_features * expand_ratio, out_channels=out_features, kernel_size=1, channels_last=channels_last),
            norm[dim](num_features=out_features, channels_last=channels_last)
        )

        # Residual
        self.residual = torch.prod(torch.tensor(stride)) == 1

    def forward(self, samples, output_dict=False, **kargs):

        # Layers
        if self.residual:
            samples = self.layers(samples) + samples
        else:
            samples = self.layers(samples)

        return {"samples": samples} if output_dict else samples

class EffNetBlock(nn.Module):

    """EfficientNet Block
    
    References: 
        https://arxiv.org/abs/1905.11946
    
    """

    def __init__(self, in_features, out_features, expand_ratio, squeeze_ratio, kernel_size, stride, act_fun, drop_rate, dim=2, channels_last=False):
        super(EffNetBlock, self).__init__()

        conv = {
            1: Conv1d,
            2: Conv2d,
            3: Conv3d
        }

        norm = {
            1: BatchNorm1d,
            2: BatchNorm2d,
            3: BatchNorm3d
        }

        # Layers
        self.layers = nn.Sequential(
            conv[dim](in_channels=in_features, out_channels=in_features * expand_ratio, kernel_size=1, channels_last=channels_last),
            norm[dim](num_features=in_features * expand_ratio, channels_last=channels_last),
            act_dict[act_fun](),
            conv[dim](in_channels=in_features * expand_ratio, out_channels=in_features * expand_ratio, kernel_size=kernel_size, stride=stride, groups=in_features * expand_ratio, channels_last=channels_last),
            norm[dim](num_features=in_features * expand_ratio, channels_last=channels_last),
            act_dict[act_fun](),
            SqueezeExciteModule(input_dim=in_features * expand_ratio, reduction_ratio=squeeze_ratio, inner_act=act_fun, dim=2, channels_last=channels_last),
            conv[dim](in_channels=in_features * expand_ratio, out_channels=out_features, kernel_size=1, channels_last=channels_last),
            norm[dim](num_features=out_features, channels_last=channels_last)
        )

        # Residual
        self.residual = torch.prod(torch.tensor(stride)) == 1

        # Dropout
        if self.residual:
            self.dropout = Dropout(drop_rate)

    def forward(self, samples, output_dict=False, **kargs):

        # Layers
        if self.residual:
            samples = self.dropout(self.layers(samples)) + samples
        else:
            samples = self.layers(samples)

        return {"samples": samples} if output_dict else samples

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

class SwitchFFN(Module):

    """ Mixture of Experts (MoE) Switch FFN Module, Single Routing Layer
    
    Reference: "SWITCH TRANSFORMERS: SCALING TO TRILLION PARAMETER MODELS WITH SIMPLE AND EFFICIENT SPARSITY" by Fedus et al.
    https://arxiv.org/abs/2101.03961

    """

    def __init__(self, num_experts, dim_model, dim_ffn, drop_rate, act_fun, inner_dropout, noise_eps=0.1):
        super(SwitchFFN, self).__init__()

        # Params
        self.num_experts = num_experts
        self.noise_eps = noise_eps        

        # Router
        self.router = Linear(dim_model, self.num_experts)

        # Experts
        #self.experts = nn.ModuleList([module(**module_params) for _ in range(self.num_experts)])
        #self.experts = nn.ModuleList([Linear(in_features=module_params["dim_model"], out_features=module_params["dim_model"]) for _ in range(self.num_experts)])

        # Modules
        self.layernorm = nn.LayerNorm(dim_model)
        self.linear1 = SwitchLinear(num_experts=num_experts, in_features=dim_model, out_features=dim_ffn)
        self.act_fun = act_dict[act_fun]()
        self.dropout1 = nn.Dropout(p=drop_rate) if inner_dropout else nn.Identity()
        self.linear2 = SwitchLinear(num_experts=num_experts, in_features=dim_ffn, out_features=dim_model)
        self.dropout2 = nn.Dropout(drop_rate)

    def compute_loss(self, router_probs, indices):

        # One Hot Indices (N, 1) -> (N, K)
        indices_one_hot = F.one_hot(indices.squeeze(dim=-1), self.num_experts).type(router_probs.dtype)

        # Router Count (K,)
        density = indices_one_hot.mean(axis=0)

        # Router Mean Prob (K,)
        density_proxy = router_probs.mean(axis=0)

        # Compute loss
        loss = (density_proxy * density).mean() * (self.num_experts ** 2)

        return loss

    def forward(self, x):

        # Shape
        tokens_shape = x.shape[:-1]

        # Flatten Tokens (..., Din) -> (N, Din)
        x = x.flatten(start_dim=0, end_dim=-2)

        # (N, Din) -> (N, K)
        router_logits = self.router(x)

        # Add noise for exploration across experts.
        if self.training:
            router_logits += torch.empty_like(router_logits).uniform_(1-self.noise_eps, 1+self.noise_eps)

        # Probabilities for each token of what expert it should be sent to.
        router_probs = router_logits.softmax(dim=-1)

        # Get Gate / Index (N, 1)
        gates, indices = router_probs.topk(k=1, dim=-1)

        # Compute load balancing loss.
        loss_switch = self.compute_loss(router_probs, indices)

        # Add Loss
        self.add_loss("switch", loss_switch)

        # Forward Experts
        #x = [self.experts[indices[token_id]](x[token_id:token_id+1]) for token_id in range(x.size(0))]

        # Stack Outputs (N, Dout)
        #x = torch.concat(x, dim=0)

        # Forward Modules
        x = self.layernorm(x)
        x = self.linear1(x, indices)
        x = self.act_fun(x)
        x = self.dropout1(x)
        x = self.linear2(x, indices)
        x = self.dropout2(x)

        # Gate Outputs
        x = x * gates

        # Reshape (N, Dout) -> (..., Dout)
        x = x.reshape(tokens_shape + x.shape[-1:])

        return x

"""
class SwitchFFN(nn.Module):

    def __init__(self, num_experts, dim_model, dim_ffn, drop_rate, act_fun, inner_dropout):
        super(SwitchFFN, self).__init__()

        # LayerNorm
        self.layernorm = nn.LayerNorm(dim_model)

        # Switch FFN
        self.switch_ffn = SwitchModule(num_experts, dim_model, FeedForwardModule, module_params={
            "dim_model": dim_model,
            "dim_ffn": dim_ffn, 
            "drop_rate": drop_rate, 
            "act_fun": act_fun, 
            "inner_dropout": inner_dropout, 
            "prenorm": False
        })

    def forward(self, x):

        # Layer Norm
        x = self.layernorm(x)

        # Switch FFN
        x = self.switch_ffn(x)

        return x
"""

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

###############################################################################
# MLP-Mixer Modules
###############################################################################

class MixerLayer(nn.Module):

    def __init__(self, dim_seq, dim_expand_seq, dim_feat, dim_expand_feat, act_fun, drop_rate):
        super(MixerLayer, self).__init__()

        # Mixer Seq
        self.mixer_seq = nn.Sequential(
            nn.LayerNorm(dim_feat),
            Transpose(1, 2),
            Linear(dim_seq, dim_expand_seq),
            act_dict[act_fun](),
            nn.Dropout(drop_rate),
            Linear(dim_expand_seq, dim_seq),
            nn.Dropout(drop_rate),
            Transpose(1, 2)
        )

        # Mixer Feat
        self.mixer_feat = nn.Sequential(
            nn.LayerNorm(dim_feat),
            Linear(dim_feat, dim_expand_feat),
            act_dict[act_fun](),
            nn.Dropout(drop_rate),
            Linear(dim_expand_feat, dim_feat),
            nn.Dropout(drop_rate),
        )

    def forward(self, x):

        # Mixer Seq
        x = x + self.mixer_seq(x)

        # Mixer Feat
        x = x + self.mixer_feat(x)

        return x

###############################################################################
# ContextNet Modules
###############################################################################

class ContextNetBlock(nn.Module):

    def __init__(self, num_layers, dim_in, dim_out, kernel_size, stride, causal, se_ratio, residual, padding):
        super(ContextNetBlock, self).__init__()

        # Conv Layers
        self.conv_layers = nn.Sequential(*[
            DepthwiseSeparableConv1d(dim_in if layer_id == 0 else dim_out, dim_out, kernel_size, stride if layer_id == num_layers - 1 else 1, causal) 
        for layer_id in range(num_layers)])

        # SE Module
        self.se_module = SqueezeAndExcitationModule(dim_out, se_ratio, "swish") if se_ratio is not None else None

        # Residual
        self.residual = nn.Sequential(
            Conv1d(dim_in, dim_out, kernel_size=1, stride=stride, groups=1, padding=padding),
            nn.BatchNorm1d(dim_out)
        ) if residual else None

        # Block Act
        self.act = Swish()

    def forward(self, x):

        # Conv Layers
        y = self.conv_layers(x)

        # SE Module
        if self.se_module is not None:
            y = self.se_module(y)

        # Residual
        if self.residual is not None:
            y = self.act(y + self.residual(x))

        return y  

class ContextNetSubsampling(nn.Module):

    def __init__(self, n_mels, dim_model, kernel_size, causal):
        super(ContextNetSubsampling, self).__init__()

        # Blocks
        self.blocks = nn.Sequential(*[ContextNetBlock(
            num_layers=1 if block_id == 0 else 5, 
            dim_in=n_mels if block_id == 0 else dim_model, 
            dim_out=dim_model, 
            kernel_size=kernel_size, 
            stride=2 if block_id in [3, 7] else 1, 
            causal=causal, 
            se_ratio=None if block_id == 0 else 8, 
            residual=False if block_id == 0 else True,
        ) for block_id in range(8)])

    def forward(self, x, x_len):

        # Blocks
        x = self.blocks(x)

        # Update Sequence Lengths
        if x_len is not None:
            x_len = (x_len - 1) // 2 + 1
            x_len = (x_len - 1) // 2 + 1

        return x, x_len

###############################################################################
# Modules
###############################################################################

class SqueezeExciteModule(nn.Module):

    """Squeeze And Excitation Module

    Args:
        input_dim: input feature dimension
        reduction_ratio: bottleneck reduction ratio
        inner_act: bottleneck inner activation function
    
    """

    def __init__(self, input_dim, reduction_ratio, inner_act="relu", dim=2, channels_last=False):
        super(SqueezeExciteModule, self).__init__()

        assert input_dim % reduction_ratio == 0
        self.conv1 = Conv1d(input_dim, input_dim // reduction_ratio, kernel_size=1)
        self.conv2 = Conv1d(input_dim // reduction_ratio, input_dim, kernel_size=1)
        self.pool = GlobalAvgPool1d(dim=-1, keepdim=True)
        self.inner_act = act_dict[inner_act]()

    def forward(self, x, x_len=None):

        # Global avg Pooling
        scale = x.mean(dim=-1, keepdim=True)

        # (B, C, 1) -> (B, C // R, 1)
        scale = self.conv1(scale)

        # Inner Act
        scale = self.inner_act(scale)

        # (B, C // R, 1) -> (B, C, 1)
        scale = self.conv2(scale)

        # Sigmoid
        scale = scale.sigmoid()

        # Scale
        x = x * scale

        return x

class IdentityProjection(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(IdentityProjection, self).__init__()

        assert output_dim > input_dim
        self.linear = Linear(input_dim, output_dim - input_dim)

    def forward(self, x):

        # (B, T, Dout - Din)
        proj = self.linear(x)

        # (B, T, Dout)
        x = torch.cat([x, proj], dim=-1)

        return x

class DepthwiseSeparableConv1d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(DepthwiseSeparableConv1d, self).__init__()

        # Layers
        self.layers = nn.Sequential(
            Conv1d(in_channels, in_channels, kernel_size, padding=padding, groups=in_channels, stride=stride),
            Conv1d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels),
            Swish()
        )

    def forward(self, x):
        return self.layers(x)

###############################################################################
# Module Dictionary
###############################################################################

module_dict = {
    "MLP": MultiLayerPerceptron,
    "CNN": ConvNeuralNetwork,
    "CTNN": ConvTransposeNeuralNetwork
}