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
    UnetBlock,
    TransformerBlock,
    CrossTransformerBlock,
    ConformerBlock,
    CrossConformerBlock,
    SwitchTransformerBlock
)

# Modules
from nnet.modules import (
    ConvNeuralNetwork,
    ConvTransposeNeuralNetwork,
    MultiLayerPerceptron,
    PatchEmbedding,
    MixerLayer
)

# Preprocessing
from nnet.preprocessing import (
    AudioPreprocessing,
    SpecAugment
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
# RNN Networks
###############################################################################

class LSTMNetwork(nn.Module):

    def __init__(self, input_embedding, dim_input, dim_model, num_layers, bidirectional):
        super(LSTMNetwork, self).__init__()

        # Input Embedding
        if input_embedding == None:
            self.input_embedding = nn.Identity()
        elif input_embedding == "Embedding":
            self.input_embedding = nn.Embedding(num_embeddings=dim_input, embedding_dim=dim_model)
        elif input_embedding == "Linear":
            self.input_embedding = Linear(in_features=dim_input, out_features=dim_model)

        # LSTM Layers
        self.layers = LSTM(
            input_size=dim_model, 
            hidden_size=dim_model, 
            num_layers=num_layers, 
            batch_first=True, 
            bidirectional=bidirectional
        )

    def forward(self, samples, lengths=None, hidden=None, output_dict=False, **kargs):


        # Input Embedding
        samples = self.input_embedding(samples)

        # Pack padded batch sequences
        if lengths is not None:
            samples = nn.utils.rnn.pack_padded_sequence(samples, lengths.cpu(), batch_first=True, enforce_sorted=False)

        # LSTM Layers
        samples, hidden = self.layers(samples, hidden)

        # Pad packed batch sequences
        if lengths is not None:
            samples, _ = nn.utils.rnn.pad_packed_sequence(samples, batch_first=True)

        # return last layer steps outputs and every layer last step hidden state
        return {"samples": samples, "hidden": hidden} if output_dict else (samples, hidden)

###############################################################################
# CNN Networks
###############################################################################

class UNet(nn.Module):

    def __init__(self, dim_input, stage_dim, layer_per_stage, kernel_size, norm, act_fun, res_drop_rate):
        super(UNet, self).__init__()

        # Unet Stages
        self.stages = None
        for stage_id in reversed(range(len(stage_dim))):
             self.stages = UnetBlock(
                 dim_in=dim_input if stage_id==0 else stage_dim[stage_id-1],
                 dim_down=stage_dim[stage_id],
                 num_layers=layer_per_stage,
                 kernel_size=kernel_size,
                 norm=norm,
                 act_fun=act_fun,
                 res_drop_rate=res_drop_rate,
                 sub_block=self.stages
             )

    def forward(self, x):

        return self.stages(x)

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

class SwitchTransformer(nn.Module):

    def __init__(self, num_experts, dim_model, mask, ff_ratio, att_params, num_blocks, pos_embedding, max_pos_encoding, drop_rate, causal, inner_dropout=False):
        super(SwitchTransformer, self).__init__()

        assert pos_embedding in [None, "Embedding", "Sinusoidal"]

        # Positional Embedding
        if pos_embedding == None:
            self.pos_embedding = None
        elif pos_embedding == "Embedding":
            self.pos_embedding = nn.Embedding(num_embeddings=max_pos_encoding, embedding_dim=dim_model)
        elif pos_embedding == "Sinusoidal":
            self.pos_embedding = SinusoidalPositionalEncoding(max_len=max_pos_encoding, dim_model=dim_model)

        # Input Dropout
        self.dropout = nn.Dropout(p=drop_rate)

        # Mask
        self.mask = mask

        # Transformer Blocks
        self.blocks = nn.ModuleList([SwitchTransformerBlock(
            num_experts=num_experts,
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

    def forward(self, x, lengths=None):

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

        return x, hidden, att_maps

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

class MLPMixerEncoder(nn.Module):

    def __init__(self, num_layers, patch_size, input_height, input_width, input_channels, dim_feat, dim_expand_feat, dim_expand_seq, act_fun, drop_rate):
        super(MLPMixerEncoder, self).__init__()

        # Num Patches
        self.num_patches = (input_height * input_width) / (patch_size**2 if isinstance(patch_size, int) else torch.prod(patch_size))

        # Assert
        assert int(self.num_patches) == self.num_patches
        self.num_patches = int(self.num_patches)

        # Patch Embedding
        self.patch_embedding = PatchEmbedding(
            input_channels=input_channels,
            patch_size=patch_size,
            embedding_dim=dim_feat
        )

        # Mixer layers
        self.layers = nn.ModuleList([MixerLayer(
            dim_seq=self.num_patches,
            dim_expand_seq=dim_expand_seq,
            dim_feat=dim_feat,
            dim_expand_feat=dim_expand_feat,
            act_fun=act_fun,
            drop_rate=drop_rate
        ) for layer_id in range(num_layers)])

    def forward(self, x):

        # Patch Embedding (B, N, D)
        x = self.patch_embedding(x)

        # MLP-Mixer Layers
        for layer in self.layers:
            x = layer(x)

        return x

class ConformerInterCTC(Conformer):

    def __init__(self, params):
        super(ConformerInterCTC, self).__init__(params)

        # Inter CTC blocks
        self.interctc_blocks = params["interctc_blocks"]
        for block_id in params["interctc_blocks"]:
            self.__setattr__(
                name="linear_expand_" + str(block_id), 
                value=nn.Linear(
                    in_features=params["dim_model"][(block_id >= torch.tensor(params.get("expand_blocks", []))).sum()] if isinstance(params["dim_model"], list) else params["dim_model"],
                    out_features=params["vocab_size"]))
            self.__setattr__(
                name="linear_proj_" + str(block_id),
                value=nn.Linear(
                    in_features=params["vocab_size"],
                    out_features=params["dim_model"][(block_id >= torch.tensor(params.get("expand_blocks", []))).sum()] if isinstance(params["dim_model"], list) else params["dim_model"]))

    def forward(self, x, x_len=None):

        # Audio Preprocessing
        x, x_len = self.preprocessing(x, x_len)

        # Spec Augment
        if self.training:
            x = self.augment(x, x_len)

        # Subsampling Module
        x, x_len = self.subsampling_module(x, x_len)

        # Padding Mask
        mask = self.padding_mask(x, x_len)

        # Transpose (B, D, T) -> (B, T, D)
        x = x.transpose(1, 2)

        # Linear Projection
        x = self.linear(x)

        # Dropout
        x = self.dropout(x)

        # Sinusoidal Positional Encodings
        if self.pos_enc is not None:
            x = x + self.pos_enc(x.size(0), x.size(1))

        # Conformer Blocks
        attentions = []
        interctc_probs = []
        for block_id, block in enumerate(self.blocks):
            x, attention, hidden = block(x, mask)
            attentions.append(attention)

            # Strided Block
            if block.stride > 1:

                # Stride Mask (B, 1, T // S, T // S)
                if mask is not None:
                    mask = mask[:, :, ::block.stride, ::block.stride]

                # Update Seq Lengths
                if x_len is not None:
                    x_len = (x_len - 1) // block.stride + 1

            # Inter CTC Block
            if block_id in self.interctc_blocks:
                interctc_prob = self.__getattr__("linear_expand_" + str(block_id))(x).softmax(dim=-1)
                interctc_probs.append(interctc_prob)
                x = x + self.__getattr__("linear_proj_" + str(block_id))(interctc_prob)

        return x, x_len, attentions, interctc_probs

###############################################################################
# Joint Networks
###############################################################################

class JointNetwork(nn.Module):

    def __init__(self, params):
        super(JointNetwork, self).__init__()

        assert params["joint_mode"] in ["concat", "sum"]

        # Model layers
        if params["dim_model"] is not None:

            # Linear Layers
            self.linear_encoder = Linear(params["dim_encoder"], params["dim_model"])
            self.linear_decoder = Linear(params["dim_decoder"], params["dim_model"])

            # Joint Mode
            if params["joint_mode"] == "concat":
                self.joint_mode = "concat"
                self.linear_joint = Linear(2 * params["dim_model"], params["vocab_size"])
            elif params["joint_mode"] == "sum":
                self.joint_mode = 'sum'
                self.linear_joint = Linear(params["dim_model"], params["vocab_size"])
        else:

            # Linear Layers
            self.linear_encoder = nn.Identity()
            self.linear_decoder = nn.Identity()

            # Joint Mode
            if params["joint_mode"] == "concat":
                self.joint_mode = "concat"
                self.linear_joint = Linear(params["dim_encoder"] + params["dim_decoder"], params["vocab_size"])
            elif params["joint_mode"] == "sum":
                assert params["dim_encoder"] == params["dim_decoder"]
                self.joint_mode = 'sum'
                self.linear_joint = Linear(params["dim_encoder"], params["vocab_size"])

        # Model Act Function
        self.act_fun = act_dict[params["act_fun"]]()

    def forward(self, inputs):

        # Unpack Inputs
        f, g = inputs["samples_enc"], inputs["samples_dec"]

        f = self.linear_encoder(f)
        g = self.linear_decoder(g)

        # Training or Eval Loss
        if self.training or (len(f.size()) == 3 and len(g.size()) == 3):
            f = f.unsqueeze(2) # (B, T, 1, D)
            g = g.unsqueeze(1) # (B, 1, U + 1, D)

            f = f.repeat([1, 1, g.size(2), 1]) # (B, T, U + 1, D)
            g = g.repeat([1, f.size(1), 1, 1]) # (B, T, U + 1, D)

        # Joint Encoder and Decoder
        if self.joint_mode == "concat":
            joint = torch.cat([f, g], dim=-1) # Training : (B, T, U + 1, 2D) / Decoding : (B, 2D)
        elif self.joint_mode == "sum":
            joint = f + g # Training : (B, T, U + 1, D) / Decoding : (B, D)

        # Act Function
        joint = self.act_fun(joint)

        # Output Linear Projection
        outputs = self.linear_joint(joint) # Training : (B, T, U + 1, V) / Decoding : (B, V)
        
        return {"samples": outputs}

###############################################################################
# Complete Networks
###############################################################################

class EfficientConformerCTCSmallLibriSpeech(nn.Module):

    def __init__(self):
        super(EfficientConformerCTCSmallLibriSpeech, self).__init__()

        # Networks
        self.networks = nn.ModuleList()

        self.networks.append(AudioPreprocessing(
            sample_rate=16000,
            n_fft=512,
            win_length_ms=25,
            hop_length_ms=10,
            n_mels=80,
            normalize=False,
            mean=-5.6501,
            std=4.2280
        ))

        self.networks.append(SpecAugment(
            mF=2,
            F=27,
            mT=5,
            pS=0.05
        ))

        self.networks.append(Unsqueeze(dim=1))

        self.networks.append(Conv2dNeuralNetwork(
            dim_input=1,
            dim_layers=[120],
            kernel_size=3,
            strides=[[2, 2]],
            act_fun="Swish",
            norm="Batch",
            drop_rate=0
        ))

        self.networks.append(Reshape(shape=(4800, -1), include_batch=False))

        self.networks.append(Transpose(1, 2))

        self.networks.append(Conformer(
            input_embedding="Linear",
            dim_input=4800,
            pos_embedding=None,
            relative_pos_enc=True,
            max_pos_encoding=10000,
            causal=False,
            num_blocks=15,
            dim_model=[120, 168, 240],
            ff_ratio=4,
            num_heads=4,
            kernel_size=15,
            drop_rate=0.1,
            conv_stride=2,
            att_stride=1,
            strided_blocks=[4, 9],
            expand_blocks=[4, 9],
            att_group_size=[3, 1, 1],
            att_kernel_size=None
        ))

        self.networks.append(MultiLayerPerceptron(
            dim_input=240,
            dim_layers=[256],
            act_fun=None,
            out_fun=None,
            norm=None,
            drop_rate=0
        ))

    def forward(self, **inputs):

        for module in self.networks:
            inputs.update(module(**inputs))

        return inputs

###############################################################################
# Networks Dictionary
###############################################################################

networks_dict = {
    "LSTM": LSTMNetwork,
    "UNet": UNet,
    "Transformer": Transformer,
    "CrossTransformer": CrossTransformer,
    "Conformer": Conformer,
    "MLPMixerEncoder": MLPMixerEncoder,
    "JointNetwork": JointNetwork,
    "EfficientConformerCTCSmallLibriSpeech": EfficientConformerCTCSmallLibriSpeech
}