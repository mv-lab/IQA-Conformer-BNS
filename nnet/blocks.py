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
    ConvNeuralNetwork,
    ConvTransposeNeuralNetwork,
    FeedForwardModule,
    AttentionModule,  
    ConvolutionModule,
    Transpose,
    SwitchFFN
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
# ResNet Blocks
###############################################################################

class ResNetV2Block(nn.Module):

    """ ResNetV2 Residual Block used by ResNet18V2 and ResNet34V2 networks.

    References: 
        https://arxiv.org/abs/1603.05027
    
    """

    def __init__(self, in_features, out_features, kernel_size, stride, act_fun, dim=2, channels_last=False):
        super(ResNetV2Block, self).__init__()

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

        # Pre Norm
        self.pre_norm = nn.Sequential(
            norm[dim](num_features=in_features, channels_last=channels_last),
            act_dict[act_fun]()
        )

        # layers
        self.layers = nn.Sequential(
            conv[dim](in_channels=in_features, out_channels=out_features, kernel_size=kernel_size, stride=stride, channels_last=channels_last, bias=False),

            norm[dim](num_features=out_features, channels_last=channels_last),
            act_dict[act_fun](),
            conv[dim](in_channels=out_features, out_channels=out_features, kernel_size=kernel_size, channels_last=channels_last),
        )

        # Pooling Block
        if torch.prod(torch.tensor(stride)) > 1:
            self.pooling = nn.MaxPool2d(kernel_size=1, stride=stride)
            self.conv = None
        
        # Projection Block
        elif in_features != out_features:
            self.pooling = None
            self.conv = conv[dim](in_channels=in_features, out_channels=out_features, kernel_size=1, channels_last=channels_last)

        # Default Block
        else:
            self.pooling = None
            self.conv = None

    def forward(self, x):

        # Pooling Block
        if self.pooling != None:
            x = self.layers(self.pre_norm(x)) + self.pooling(x)

        # Projection Block
        elif self.conv != None:
            x = self.pre_norm(x)
            x = self.layers(x) + self.conv(x)

        # Default Block
        else:
            x = self.layers(self.pre_norm(x)) + x

        return x

class ResNetV2BottleneckBlock(nn.Module):

    """ ResNetV2 Bottleneck Residual Block used by ResNet50V2, ResNet101V2 and ResNet152V2 networks.

    References: 
        https://arxiv.org/abs/1603.05027

    """

    def __init__(self, in_features, out_features, bottleneck_ratio, kernel_size, stride, act_fun, dim=2, channels_last=False):
        super(ResNetV2BottleneckBlock, self).__init__()

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

        # Assert
        assert in_features % bottleneck_ratio == 0

        # Pre Norm
        self.pre_norm = nn.Sequential(
            norm[dim](num_features=in_features, channels_last=channels_last),
            act_dict[act_fun]()
        )

        # layers
        self.layers = nn.Sequential(
            conv[dim](in_channels=in_features, out_channels=in_features // bottleneck_ratio, kernel_size=1, channels_last=channels_last, bias=False),

            norm[dim](num_features=in_features // bottleneck_ratio, channels_last=channels_last),
            act_dict[act_fun](),
            conv[dim](in_channels=in_features // bottleneck_ratio, out_channels=in_features // bottleneck_ratio, kernel_size=kernel_size, stride=stride, channels_last=channels_last, bias=False),
            
            norm[dim](num_features=in_features // bottleneck_ratio, channels_last=channels_last),
            act_dict[act_fun](),
            conv[dim](in_channels=in_features // bottleneck_ratio, out_channels=out_features, kernel_size=1, channels_last=channels_last)
        )

        # Pooling Block
        if torch.prod(torch.tensor(stride)) > 1:
            self.pooling = nn.MaxPool2d(kernel_size=1, stride=stride)
            self.conv = None
        
        # Projection Block
        elif in_features != out_features:
            self.pooling = None
            self.conv = conv[dim](in_channels=in_features, out_channels=out_features, kernel_size=1, channels_last=channels_last)

        # Default Block
        else:
            self.pooling = None
            self.conv = None

    def forward(self, x):

        # Pooling Block
        if self.pooling != None:
            x = self.layers(self.pre_norm(x)) + self.pooling(x)

        # Projection Block
        elif self.conv != None:
            x = self.pre_norm(x)
            x = self.layers(x) + self.conv(x)

        # Default Block
        else:
            x = self.layers(self.pre_norm(x)) + x

        return x

###############################################################################
# Unet Blocks
###############################################################################

class UnetBlock(nn.Module):

    def __init__(self, dim_in, dim_down, num_layers, kernel_size, norm, act_fun, res_drop_rate, sub_block=None):
        super(UnetBlock, self).__init__()

        # CNN Down
        cnn_down = ConvNeuralNetwork(
            dim_input=dim_in,
            dim_layers=[dim_down for _ in range(num_layers)],
            kernel_size=kernel_size,
            strides=[2] + [1 for _ in range(num_layers - 1)],
            norm=norm,
            act_fun=act_fun,
            drop_rate=0
        )

        # CNN Up
        cnn_up = ConvTransposeNeuralNetwork(
            dim_input=dim_down if sub_block == None else  2 * dim_down,
            dim_layers=[dim_down for _ in range(0 if sub_block == None else num_layers - 1)] + [dim_in],
            kernel_size=kernel_size,
            strides=[1 for _ in range(0 if sub_block == None else num_layers - 1)] + [2],
            norm=norm,
            act_fun=act_fun,
            drop_rate=0,
            padding=1,
            output_padding=[0 for _ in range(0 if sub_block == None else num_layers - 1)] + [0]
        )

        # Sub Block
        if sub_block != None:
            self.layers = nn.Sequential(
                cnn_down,
                sub_block,
                cnn_up,
                nn.Dropout(res_drop_rate)
            )
        else:
            self.layers = nn.Sequential(
                cnn_down,
                cnn_up
            )

    def forward(self, x):

        return torch.cat([x, self.layers(x)], dim=1)

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

class SwitchTransformerBlock(nn.Module):

    def __init__(self, num_experts, dim_model, att_params, max_pos_encoding, ff_ratio=4, Pdrop=0.1, causal=False, inner_dropout=False, act_fun="ReLU"):
        super(SwitchTransformerBlock, self).__init__()

        # Muti-Head Self-Attention Module
        self.self_att_module = AttentionModule(
            dim_model=dim_model,
            att_params=att_params,
            drop_rate=Pdrop,
            max_pos_encoding=max_pos_encoding,
            causal=causal
        )

        # Feed Forward Module
        self.ff_module = SwitchFFN(
            num_experts=num_experts,
            dim_model=dim_model, 
            dim_ffn=dim_model * ff_ratio, 
            drop_rate=Pdrop, 
            act_fun=act_fun,
            inner_dropout=inner_dropout
        )

    def forward(self, x, mask=None, hidden=None):

        # Muti-Head Self-Attention Module
        x_att, attention, hidden = self.self_att_module(x, mask, hidden)
        x = x + x_att

        # Feed Forward Module
        x = x + self.ff_module(x)

        return x, attention, hidden

###############################################################################
# Block Dictionary
###############################################################################

block_dict = {
    "TransformerBlock": TransformerBlock,
    "CrossTransformerBlock": CrossTransformerBlock,
    "ConformerBlock": ConformerBlock
}