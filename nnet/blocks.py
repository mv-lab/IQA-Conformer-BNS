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

# Neural Nets
from nnet.modules import (
    FeedForwardModule,
    AttentionModule,  
    ConvolutionModule,
    Transpose
)

# Layers
from nnet.layers import (
    layer_dict,
    Conv1d,
    Conv2d,
    Conv3d
)

from nnet.normalizations import (
    BatchNorm1d,
    BatchNorm2d,
    BatchNorm3d
)

from nnet.activations import (
    act_dict
)

###############################################################################
# Transformer Blocks
###############################################################################

class ConformerBlock(nn.Module):

    def __init__(
        self, 
        dim_model, 
        dim_expand, 
        ff_ratio,
        att_params,
        drop_rate, 
        max_pos_encoding, 
        conv_stride,
        att_stride,
        causal,
        conv_params,
        inner_dropout=True,
        act_fun="Swish"
    ):
        super(ConformerBlock, self).__init__()

        assert conv_params["class"] in ["Conv1d", "Conv2d", "Conv3d", "ConvTranspose1d", "ConvTranspose2d", "ConvTranspose3d"]

        # Feed Forward Module 1
        self.ff_module1 = FeedForwardModule(
            dim_model=dim_model, 
            dim_ffn=dim_model * ff_ratio,
            drop_rate=drop_rate, 
            act_fun=act_fun,
            inner_dropout=inner_dropout
        )

        # Multi-Head Self-Attention Module
        self.self_att_module = AttentionModule(
            dim_model=dim_model, 
            att_params=att_params, 
            drop_rate=drop_rate, 
            max_pos_encoding=max_pos_encoding, 
            causal=causal
        )

        # Convolution Module
        self.conv_module = ConvolutionModule(
            dim_model=dim_model,
            dim_expand=dim_expand,
            drop_rate=drop_rate, 
            stride=conv_stride,
            act_fun=act_fun,
            conv_params=conv_params,
            channels_last=True
        )

        # Feed Forward Module 2
        self.ff_module2 = FeedForwardModule(
            dim_model=dim_expand, 
            dim_ffn=dim_expand * ff_ratio,
            drop_rate=drop_rate, 
            act_fun=act_fun,
            inner_dropout=inner_dropout
        )

        # Block Norm
        self.norm = nn.LayerNorm(dim_expand)

        # Transposed Block
        transposed_block = "Transpose" in conv_params["class"]

        # Attention Residual # Not Updated
        self.att_res = nn.Sequential(
            Transpose(1, 2),
            nn.MaxPool1d(kernel_size=1, stride=att_stride),
            Transpose(1, 2)
        ) if att_stride > 1 else nn.Identity()

        # Convolution Residual
        if dim_model != dim_expand: # Pointwise Conv Residual
            if transposed_block:
                self.conv_res = layer_dict[conv_params["class"]](dim_model, dim_expand, kernel_size=1, stride=conv_stride, channels_last=True, output_padding=conv_params["params"].get("output_padding", 0))
            else:
                self.conv_res = layer_dict[conv_params["class"]](dim_model, dim_expand, kernel_size=1, stride=conv_stride, channels_last=True)
        elif conv_stride > 1: # Pooling / Upsample Residual
            if transposed_block:
                self.conv_res = layer_dict[conv_params["class"].replace("ConvTranspose", "Upsample")](scale_factor=conv_stride)
            else:
                self.conv_res = layer_dict[conv_params["class"].replace("Conv", "MaxPool")](kernel_size=1, stride=conv_stride)
        else: # Identity Residual
            self.conv_res = nn.Identity()

        # Bloc Stride
        self.stride = conv_stride * att_stride

    def forward(self, x, mask=None, hidden=None):

        # FFN Module 1
        x = x + 1/2 * self.ff_module1(x)

        # MHSA Module
        x_att, attention, hidden = self.self_att_module(x, mask=mask, hidden=hidden)
        x = self.att_res(x) + x_att

        # Conv Module
        x = self.conv_res(x) + self.conv_module(x)

        # FFN Module 2
        x = x + 1/2 * self.ff_module2(x)

        # Block Norm
        x = self.norm(x)

        return x, attention, hidden

class CrossConformerBlock(nn.Module):

    def __init__(
        self, 
        dim_model, 
        dim_expand, 
        ff_ratio,
        att_params,
        cross_att_params,
        drop_rate, 
        max_pos_encoding, 
        conv_stride,
        att_stride,
        causal,
        conv_params,
        inner_dropout=True,
        act_fun="Swish"
    ):
        super(CrossConformerBlock, self).__init__()

        assert conv_params["class"] in ["Conv1d", "Conv2d", "Conv3d", "ConvTranspose1d", "ConvTranspose2d", "ConvTranspose3d"]

        # Feed Forward Module 1
        self.ff_module1 = FeedForwardModule(
            dim_model=dim_model, 
            dim_ffn=dim_model * ff_ratio,
            drop_rate=drop_rate, 
            act_fun=act_fun,
            inner_dropout=inner_dropout
        )

        # Multi-Head Self-Attention Module
        self.self_att_module = AttentionModule(
            dim_model=dim_model, 
            att_params=att_params, 
            drop_rate=drop_rate, 
            max_pos_encoding=max_pos_encoding, 
            causal=causal
        )

        # Muti-Head Cross-Attention Module
        self.cross_attention_module = AttentionModule(
            dim_model=dim_model,
            att_params=cross_att_params,
            drop_rate=drop_rate,
            max_pos_encoding=max_pos_encoding,
            causal=causal
        )

        # Convolution Module
        self.conv_module = ConvolutionModule(
            dim_model=dim_model,
            dim_expand=dim_expand,
            drop_rate=drop_rate, 
            stride=conv_stride,
            act_fun=act_fun,
            conv_params=conv_params,
            channels_last=True
        )

        # Feed Forward Module 2
        self.ff_module2 = FeedForwardModule(
            dim_model=dim_expand, 
            dim_ffn=dim_expand * ff_ratio,
            drop_rate=drop_rate, 
            act_fun=act_fun,
            inner_dropout=inner_dropout
        )

        # Block Norm
        self.norm = nn.LayerNorm(dim_expand)

        # Attention Residual # Not Updated
        self.att_res = nn.Sequential(
            Transpose(1, 2),
            nn.MaxPool1d(kernel_size=1, stride=att_stride),
            Transpose(1, 2)
        ) if att_stride > 1 else nn.Identity()

        # Convolution Residual
        if dim_model != dim_expand: # Pointwise Conv Residual
            self.conv_res = layer_dict[conv_params["class"].replace("Transpose", "")](dim_model, dim_expand, kernel_size=1, stride=conv_stride, channels_last=True)
        elif conv_stride > 1: # Pooling Residual
            self.conv_res = layer_dict[conv_params["class"].replace("Transpose", "").replace("Conv", "MaxPool")](kernel_size=1, stride=conv_stride)
        else: # Identity Residual
            self.conv_res = nn.Identity()

        # Bloc Stride
        self.stride = conv_stride * att_stride

    def forward(self, x, x_enc, mask=None, mask_enc=None, hidden=None):

        # FFN Module 1
        x = x + 1/2 * self.ff_module1(x)

        # MHSA Module
        x_att, self_att_map, hidden = self.self_att_module(x, mask=mask, hidden=hidden)
        x = self.att_res(x) + x_att

        # Muti-Head Cross-Attention Module
        x_att, cros_att_map, _ = self.cross_attention_module(x, x_cross=x_enc, mask=mask_enc)
        x = x + x_att

        # Conv Module
        x = self.conv_res(x) + self.conv_module(x)

        # FFN Module 2
        x = x + 1/2 * self.ff_module2(x)

        # Block Norm
        x = self.norm(x)

        return x, {"self": self_att_map, "cross": cros_att_map}, hidden

class TransformerBlock(nn.Module):

    def __init__(self, dim_model, att_params, max_pos_encoding, ff_ratio=4, Pdrop=0.1, causal=False, inner_dropout=False, act_fun="ReLU"):
        super(TransformerBlock, self).__init__()

        # Muti-Head Self-Attention Module
        self.self_att_module = AttentionModule(
            dim_model=dim_model,
            att_params=att_params,
            drop_rate=Pdrop,
            max_pos_encoding=max_pos_encoding,
            causal=causal
        )

        # Feed Forward Module
        self.ff_module = FeedForwardModule(
            dim_model=dim_model, 
            dim_ffn=dim_model * ff_ratio, 
            drop_rate=Pdrop, 
            act_fun=act_fun,
            inner_dropout=inner_dropout
        )

    def forward(self, x, mask=None, hidden=None):

        # Muti-Head Self-Attention Module
        x_att, attention, hidden = self.self_att_module(x, mask=mask, hidden=hidden)
        x = x + x_att

        # Feed Forward Module
        x = x + self.ff_module(x)

        return x, attention, hidden

class CrossTransformerBlock(nn.Module):

    def __init__(self, dim_model, ff_ratio, att_params, cross_att_params, drop_rate, max_pos_encoding, causal, inner_dropout, act_fun="ReLU"):
        super(CrossTransformerBlock, self).__init__()

        # Muti-Head Self-Attention Module
        self.self_att_module = AttentionModule(
            dim_model=dim_model,
            att_params=att_params,
            drop_rate=drop_rate,
            max_pos_encoding=max_pos_encoding,
            causal=causal
        )

        # Muti-Head Cross-Attention Module
        self.cross_attention_module = AttentionModule(
            dim_model=dim_model,
            att_params=cross_att_params,
            drop_rate=drop_rate,
            max_pos_encoding=max_pos_encoding,
            causal=causal
        )

        # Feed Forward Module
        self.feed_forward_module = FeedForwardModule(
            dim_model=dim_model, 
            dim_ffn=dim_model * ff_ratio, 
            drop_rate=drop_rate, 
            act_fun=act_fun,
            inner_dropout=inner_dropout
        )

    def forward(self, x, x_enc, mask=None, mask_enc=None, hidden=None):

        # Muti-Head Self-Attention Module
        x_att, self_att_map, hidden = self.self_att_module(x, mask=mask, hidden=hidden)
        x = x + x_att

        # Muti-Head Cross-Attention Module
        x_att, cros_att_map, _ = self.cross_attention_module(x, x_cross=x_enc, mask=mask_enc)
        x = x + x_att

        # Feed Forward Module
        x = x + self.feed_forward_module(x)

        return x, {"self": self_att_map, "cross": cros_att_map}, hidden
