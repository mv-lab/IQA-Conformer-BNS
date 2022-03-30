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

# Blocks
from nnet.blocks import (
    TransformerBlock,
    CrossTransformerBlock,
    ConformerBlock,
    CrossConformerBlock,
)

# Layers
from nnet.layers import (
    Linear,
    Conv2d,
    LSTM,
    Embedding,
    Transpose,
    Reshape,
    Unsqueeze
)

# Positional Encodings and Masks
from nnet.attentions import (
    SinusoidalPositionalEncoding,
    PaddingMask,
    Mask
)

# Activation Functions
from nnet.activations import (
    act_dict
)

###############################################################################
# Transformer Networks
###############################################################################

class Transformer(nn.Module):

    def __init__(self, dim_model, num_blocks, att_params={"class": "MultiHeadAttention", "num_heads": 4}, ff_ratio=4, drop_rate=0.1, causal=False, pos_embedding=None, max_pos_encoding=None, mask=None, inner_dropout=False):
        super(Transformer, self).__init__()

        # Positional Embedding
        if pos_embedding == None:
            self.pos_embedding = None
        elif pos_embedding == "Embedding":
            self.pos_embedding = nn.Embedding(num_embeddings=max_pos_encoding, embedding_dim=dim_model)
        elif pos_embedding == "Sinusoidal":
            self.pos_embedding = SinusoidalPositionalEncoding(max_len=max_pos_encoding, dim_model=dim_model)
        else:
            raise Exception("Unknown Positional Embedding")

        # Input Dropout
        self.dropout = nn.Dropout(p=drop_rate)

        # Mask
        self.mask = mask

        # Transformer Blocks
        self.blocks = nn.ModuleList([TransformerBlock(
            dim_model=dim_model,
            ff_ratio=ff_ratio,
            att_params=att_params,
            Pdrop=drop_rate,
            max_pos_encoding=max_pos_encoding,
            causal=causal,
            inner_dropout=inner_dropout
        ) for block_id in range(num_blocks)])

        # LayerNorm
        self.layernorm = nn.LayerNorm(
            normalized_shape=dim_model
        )

    def forward(self, x, lengths=None, return_hidden=False, return_att_maps=False):

        # Pos Embedding
        if isinstance(self.pos_embedding, Embedding):
            x += self.pos_embedding(torch.arange(torch.prod(torch.tensor(x.shape[1:-1])), device=x.device).reshape(x.shape[1:-1])) 
        elif isinstance(self.pos_embedding, SinusoidalPositionalEncoding):
            x += self.pos_embedding(seq_len=torch.prod(torch.tensor(x.shape[1:-1]))).reshape(x.shape[1:])

        # Input Dropout
        x = self.dropout(x)

        # Mask (1 or B, 1, N, N)
        if self.mask != None:
            mask = self.mask(x, lengths)
        else:
            mask = None

        # Transformer Blocks
        att_maps = []
        for block in self.blocks:
            x, att_map, hidden = block(x, mask=mask, hidden=None)
            att_maps.append(att_map)

        # LayerNorm
        x = self.layernorm(x)

        return (x, hidden, att_maps) if return_hidden and return_att_maps else (x, hidden) if return_hidden else (x, att_maps) if return_att_maps else x

class CrossTransformer(nn.Module):

    def __init__(self, dim_model, num_blocks, att_params={"class": "MultiHeadAttention", "num_heads": 4}, cross_att_params={"class": "MultiHeadAttention", "num_heads": 4}, ff_ratio=4, drop_rate=0.1, causal=False, pos_embedding=None, max_pos_encoding=None, mask=None, mask_enc=None, inner_dropout=False):
        super(CrossTransformer, self).__init__()

        # Positional Embedding
        if pos_embedding == None:
            self.pos_embedding = None
        elif pos_embedding == "Embedding":
            self.pos_embedding = nn.Embedding(num_embeddings=max_pos_encoding, embedding_dim=dim_model)
        elif pos_embedding == "Sinusoidal":
            self.pos_embedding = SinusoidalPositionalEncoding(max_len=max_pos_encoding, dim_model=dim_model)
        else:
            raise Exception("Unknown Positional Embedding")

        # Input Dropout
        self.dropout = nn.Dropout(drop_rate)

        # Masks
        self.mask = mask
        self.mask_enc = mask_enc

        # Transformer Blocks
        self.blocks = nn.ModuleList([CrossTransformerBlock(
            dim_model=dim_model,
            ff_ratio=ff_ratio,
            att_params=att_params,
            cross_att_params=cross_att_params,
            drop_rate=drop_rate,
            max_pos_encoding=max_pos_encoding,
            causal=causal,
            inner_dropout=inner_dropout
        ) for block_id in range(num_blocks)])

        # LayerNorm
        self.layernorm = nn.LayerNorm(normalized_shape=dim_model)

    def forward(self, x, x_enc, lengths=None, lengths_enc=None, return_hidden=False, return_att_maps=False):

        # Pos Embedding
        if isinstance(self.pos_embedding, Embedding):
            x += self.pos_embedding(torch.arange(torch.prod(torch.tensor(x.shape[1:-1])), device=x.device).reshape(x.shape[1:-1])) 
        elif isinstance(self.pos_embedding, SinusoidalPositionalEncoding):
            x += self.pos_embedding(seq_len=torch.prod(torch.tensor(x.shape[1:-1]))).reshape(x.shape[1:])

        # Input Dropout
        x = self.dropout(x)

        # Mask (1 or B, 1, N, N)
        if self.mask != None:
            mask = self.mask(x, lengths)
        else:
            mask = None

        # Mask Encoder (B, 1, 1, Tenc)
        if self.mask_enc != None:
            mask_enc = self.mask_enc(x, lengths_enc)
        else:
            mask_enc = None

        # Transformer Blocks
        att_maps = []
        for block in self.blocks:
            x, att_map, hidden = block(x, x_enc, mask=mask, mask_enc=mask_enc, hidden=None)
            att_maps.append(att_map)

        # LayerNorm
        x = self.layernorm(x)

        return (x, hidden, att_maps) if return_hidden and return_att_maps else (x, hidden) if return_hidden else (x, att_maps) if return_att_maps else x

class Conformer(nn.Module):

    def __init__(self, dim_model, num_blocks, att_params={"class": "MultiHeadAttention", "num_heads": 4}, conv_params={"class": "Conv1d", "params": {"padding": "same", "kernel_size": 31}}, ff_ratio=4, drop_rate=0.1, causal=False, pos_embedding=None, max_pos_encoding=None, mask=None, conv_stride=1, att_stride=1):
        super(Conformer, self).__init__()
        
        # Positional Embedding
        if pos_embedding == None:
            self.pos_embedding = None
        elif pos_embedding == "Embedding":
            self.pos_embedding = nn.Embedding(num_embeddings=max_pos_encoding, embedding_dim=dim_model[0] if isinstance(dim_model, list) else dim_model)
        elif pos_embedding == "Sinusoidal":
            self.pos_embedding = SinusoidalPositionalEncoding(max_len=max_pos_encoding, dim_model=dim_model[0] if isinstance(dim_model, list) else dim_model)
        else:
            raise Exception("Unknown Positional Embedding")

        # Input Dropout
        self.dropout = nn.Dropout(p=drop_rate)

        # Mask
        self.mask = mask

        # Conformer Stages
        self.blocks = nn.ModuleList()
        for stage_id in range(len(num_blocks)):

            # Conformer Blocks
            for block_id in range(num_blocks[stage_id]):

                # Transposed Block
                transposed_block = "Transpose" in conv_params["class"]

                # Downsampling Block
                down_block = ((block_id == 0) and (stage_id > 0)) if transposed_block else ((block_id == num_blocks[stage_id] - 1) and (stage_id < len(num_blocks) - 1))

                # Block
                self.blocks.append(ConformerBlock(
                    dim_model=dim_model[stage_id - (1 if transposed_block else 0)] if down_block else dim_model[stage_id],
                    dim_expand=dim_model[stage_id + (0 if transposed_block else 1)] if down_block else dim_model[stage_id],
                    ff_ratio=ff_ratio,
                    drop_rate=drop_rate,
                    att_params=att_params[stage_id] if isinstance(att_params, list) else att_params,
                    max_pos_encoding=max_pos_encoding,
                    conv_stride=1 if not down_block else conv_stride[stage_id] if isinstance(conv_stride, list) else conv_stride,
                    att_stride=att_stride if down_block else 1,
                    causal=causal,
                    conv_params=conv_params[stage_id] if isinstance(conv_params, list) else conv_params,
                ))

    def forward(self, x, lengths=None, return_lengths=False, return_hidden=False, return_att_maps=False):

        # Pos Embedding
        if isinstance(self.pos_embedding, Embedding):
            x += self.pos_embedding(torch.arange(torch.prod(torch.tensor(x.shape[1:-1])), device=x.device).reshape(x.shape[1:-1])) 
        elif isinstance(self.pos_embedding, SinusoidalPositionalEncoding):
            x += self.pos_embedding(seq_len=torch.prod(torch.tensor(x.shape[1:-1]))).reshape(x.shape[1:])

        # Dropout
        x = self.dropout(x)

        # Mask (1 or B, 1, N, N)
        if self.mask != None:
            mask = self.mask(x, lengths)
        else:
            mask = None

        # Conformer Blocks
        att_maps = []
        for block in self.blocks:
            x, attention, hidden = block(x, mask=mask, hidden=None)
            att_maps.append(attention)

            # Strided Block
            # ! Adapt to multidimensional data
            """
            if block.stride > 1:

                # Stride Mask (1 or B, 1, T // S, T // S)
                if mask is not None:
                    mask = mask[:, :, ::block.stride, ::block.stride]

                # Update Seq Lengths
                if lengths is not None:
                    lengths = torch.div(lengths - 1, block.stride, rounding_mode='floor') + 1
            """

        # Format Outputs
        if return_lengths or return_hidden or return_att_maps:
            outputs = (x,)
            if return_lengths:
                outputs += (lengths,)
            if return_hidden:
                outputs += (hidden,)
            if return_att_maps:
                outputs += (att_maps,)
        else:
            outputs = x

        return outputs

class CrossConformer(nn.Module):

    def __init__(self, dim_model, num_blocks, att_params={"class": "MultiHeadAttention", "num_heads": 4}, cross_att_params={"class": "MultiHeadAttention", "num_heads": 4}, conv_params={"class": "Conv1d", "params": {"padding": "same", "kernel_size": 31}}, ff_ratio=4, drop_rate=0.1, causal=False, pos_embedding=None, max_pos_encoding=None, mask=None, mask_enc=None, conv_stride=1, att_stride=1):
        super(CrossConformer, self).__init__()
        
        # Positional Embedding
        if pos_embedding == None:
            self.pos_embedding = None
        elif pos_embedding == "Embedding":
            self.pos_embedding = Embedding(num_embeddings=max_pos_encoding, embedding_dim=dim_model[0] if isinstance(dim_model, list) else dim_model)
        elif pos_embedding == "Sinusoidal":
            self.pos_embedding = SinusoidalPositionalEncoding(max_len=max_pos_encoding, dim_model=dim_model[0] if isinstance(dim_model, list) else dim_model)
        else:
            raise Exception("Unknown Positional Embedding")

        # Input Dropout
        self.dropout = nn.Dropout(p=drop_rate)

        # Mask
        self.mask = mask
        self.mask_enc = mask_enc

        # Conformer Stages
        self.blocks = nn.ModuleList()
        for stage_id in range(len(num_blocks)):

            # Conformer Blocks
            for block_id in range(num_blocks[stage_id]):

                # Downsampling Block
                down_block = (block_id == num_blocks[stage_id] - 1) and (stage_id < len(num_blocks) - 1)

                # Block
                self.blocks.append(CrossConformerBlock(
                    dim_model=dim_model[stage_id],
                    dim_expand=dim_model[stage_id + 1] if down_block else dim_model[stage_id],
                    ff_ratio=ff_ratio,
                    drop_rate=drop_rate,
                    att_params=att_params[stage_id] if isinstance(att_params, list) else att_params,
                    cross_att_params=cross_att_params[stage_id] if isinstance(cross_att_params, list) else cross_att_params,
                    max_pos_encoding=max_pos_encoding,
                    conv_stride=1 if not down_block else conv_stride[stage_id] if isinstance(conv_stride, list) else conv_stride,
                    att_stride=att_stride if down_block else 1,
                    causal=causal,
                    conv_params=conv_params[stage_id] if isinstance(conv_params, list) else conv_params,
                ))

    def forward(self, x, x_enc, lengths=None, lengths_enc=None, return_lengths=False, return_hidden=False, return_att_maps=False):

        # Pos Embedding
        if isinstance(self.pos_embedding, Embedding):
            x += self.pos_embedding(torch.arange(torch.prod(torch.tensor(x.shape[1:-1])), device=x.device).reshape(x.shape[1:-1])) 
        elif isinstance(self.pos_embedding, SinusoidalPositionalEncoding):
            x += self.pos_embedding(seq_len=torch.prod(torch.tensor(x.shape[1:-1]))).reshape(x.shape[1:])

        # Dropout
        x = self.dropout(x)

        # Mask (1 or B, 1, N, N)
        if self.mask != None:
            mask = self.mask(x, lengths)
        else:
            mask = None

        # Mask Encoder (B, 1, 1, Tenc)
        if self.mask_enc != None:
            mask_enc = self.mask_enc(x, lengths_enc)
        else:
            mask_enc = None

        # Conformer Blocks
        att_maps = []
        for block in self.blocks:
            x, attention, hidden = block(x, x_enc, mask=mask, mask_enc=mask_enc, hidden=None)
            att_maps.append(attention)


        # Format Outputs
        if return_lengths or return_hidden or return_att_maps:
            outputs = (x,)
            if return_lengths:
                outputs += (lengths,)
            if return_hidden:
                outputs += (hidden,)
            if return_att_maps:
                outputs += (att_maps,)
        else:
            outputs = x

        return outputs