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

# NeuralNets
from nnet.models import Classifier
from nnet.networks import Transformer, Conformer, MLPMixerEncoder
from nnet.blocks import ResNetV2BottleneckBlock
from nnet.layers import Conv2d, Conv3d, Linear, Embedding, GlobalAvgPool1d, GlobalAvgPool2d, GlobalAvgPool3d
from nnet.normalizations import BatchNorm2d

class ResNetV2(Classifier):

    """ ResNetV2 (ResNet50V2, ResNet101V2, ResNet152V2)

    Models: 224 x 224
    ResNet50V2: 25,568,360 Params
    ResNet101V2: 44,577,896 Params
    Resnet152V2: 60,236,904 Params

    Reference: "Identity Mappings in Deep Residual Networks" by He et al.
    https://arxiv.org/abs/1603.05027

    """

    def __init__(self, dim_input=3, dim_output=1000, model="ResNet50V2"):
        super(ResNetV2, self).__init__(name=model)

        assert model in ["ResNet50V2", "ResNet101V2", "ResNet152V2"]

        if model == "ResNet50V2":
            num_blocks = [3, 4, 6, 3]
        elif model == "ResNet101V2":
            num_blocks = [3, 4, 23, 3]
        elif model == "ResNet152V2":
            num_blocks = [3, 8, 36, 3]

        dim_model = [256, 512, 1024, 2048]

        self.stem = nn.Sequential(
            Conv2d(in_channels=dim_input, out_channels=64, kernel_size=(7, 7), stride=(2, 2)),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        )

        # Blocks
        self.blocks = nn.ModuleList()
        for stage_id in range(4):

            for block_id in range(num_blocks[stage_id]):

                # in_features
                if block_id == 0:
                    if stage_id == 0:
                        in_features = 64
                    else:
                        in_features = dim_model[stage_id-1]
                else:
                    in_features = dim_model[stage_id]

                # stride
                if block_id == num_blocks[stage_id] - 1:
                    stride = (2, 2)
                else:
                    stride = (1, 1)

                # bottleneck_ratio
                if block_id == 0:
                    if stage_id == 0:
                        bottleneck_ratio = 1
                    else:
                        bottleneck_ratio = 2
                else:
                    bottleneck_ratio = 4

                self.blocks.append(ResNetV2BottleneckBlock(
                    in_features=in_features,
                    out_features=dim_model[stage_id],
                    bottleneck_ratio=bottleneck_ratio,
                    kernel_size=(3, 3),
                    stride=stride,
                    act_fun="ReLU"
                ))

        # Head
        self.head = nn.Sequential(
            BatchNorm2d(num_features=2048),
            nn.ReLU(),
            GlobalAvgPool2d(),
            Linear(in_features=2048, out_features=dim_output)
        )

    def forward(self, x):

        # (B, Din, H, W) -> (B, D0, H//4, W//4)
        x = self.stem(x)

        # (B, D0, H//4, W//4) -> (B, D4, H//32, W//32)
        for block in self.blocks:
            x = block(x)

        # (B, D4, H//32, W//32) -> (B, Dout)
        x = self.head(x)

        return x

class ViT(Classifier):

    """ Vision Transformer (ViT-B/32, ViT-B/16, ViT-L/32, ViT-L/16, ViT-H/14)

    Models: 224 x 224
    ViT-B/32: 88 MParams    8.7 GFlops
    ViT-B/16: 86 MParams    35 GFlops
    ViT-L/32: 306 MParams   30 GFlops
    ViT-L/16: 304 MParams   122 GFlops
    ViT-H/14: 632 MParams   333 GFlops

    Models: 384 x 384
    ViT-B/32: 88 MParams    26 GFlops
    ViT-B/16: 86 MParams    110 GFlops
    ViT-L/32: 306 MParams   90 GFlops
    ViT-L/16: 304 MParams   381 GFlops
    ViT-H/14: 632 MParams   1005 GFlops
    
    Reference: "AN IMAGE IS WORTH 16X16 WORDS: TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE" by Dosovitskiy et al.
    https://arxiv.org/abs/2010.11929
    
    """

    def __init__(self, input_shape=(3, 224, 224), dim_output=1000, model="ViT-B/16"):
        super(ViT, self).__init__(name=model)

        assert model in ["ViT-B/32", "ViT-B/16", "ViT-L/32", "ViT-L/16", "ViT-H/14"]

        if model == "ViT-B/32":
            dim_model = 768
            patch_size = 32
            num_heads = 12
            num_blocks = 12
        elif model == "ViT-B/16":
            dim_model = 768
            patch_size = 16
            num_heads = 12
            num_blocks = 12
        elif model == "ViT-L/32":
            dim_model = 1024
            patch_size = 32
            num_heads = 16
            num_blocks = 24
        elif model == "ViT-L/16":
            dim_model = 1024
            patch_size = 16
            num_heads = 16
            num_blocks = 24
        elif model == "ViT-H/14":
            dim_model = 1280
            patch_size = 14
            num_heads = 16
            num_blocks = 32

        self.patch_emb = Conv2d(
            in_channels=input_shape[0],
            out_channels=dim_model,
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size)
        )

        self.class_embedding = Embedding(num_embeddings=1, embedding_dim=dim_model)

        self.transformer = Transformer(
            dim_model=dim_model,
            mask=None,
            ff_ratio=4,
            att_params={"class":"MultiHeadAttention", "num_heads": num_heads},
            num_blocks=num_blocks,
            pos_embedding="Embedding",
            max_pos_encoding=(input_shape[1] * input_shape[2]) // (patch_size * patch_size) + 1,
            drop_rate=0.1,
            causal=False,
            inner_dropout=True
        )

        self.head = Linear(
            in_features=dim_model,
            out_features=dim_output,
        )

    def forward(self, x):

        # (B, C, H, W) -> (B, H//P, W//P, D)
        x = self.patch_emb(x).permute(0, 2, 3, 1)

        # (B, H//P, W//P, D) -> (B, N, D)
        x = x.flatten(start_dim=1, end_dim=2)

        # Class Token (B, 1, D)
        class_token = self.class_embedding(x.new_zeros(x.size(0), dtype=torch.int)).unsqueeze(1)

        # (B, N, D) -> (B, 1 + N, D)
        x = torch.cat([class_token, x], dim=1)

        # Transformer
        x, _, _ = self.transformer(x)

        # (B, N, D) -> (B, Dout)
        x = self.head(x[:, 0])

        return x

class MLPMixer(Classifier):

    """ MLP-Mixer (Mixer-S/32, Mixer-S/16, Mixer-B/32, Mixer-B/16, Mixer-L/32, Mixer-L/16, Mixer-H/14)

    Models: 224 x 224
    Mixer-S/32: 19 MParams
    Mixer-S/16: 18 MParams
    Mixer-B/32: 60 MParams
    Mixer-B/16: 59 MParams
    Mixer-L/32: 206 MParams
    Mixer-L/16: 207 MParams
    Mixer-H/14: 431 MParams

    Reference: "MLP-Mixer: An all-MLP Architecture for Vision" by Tolstikhin et al.
    https://arxiv.org/abs/2105.01601
    
    """

    def __init__(self, input_shape=(3, 224, 224), dim_output=1000, model="Mixer-S/32"):
        super(MLPMixer, self).__init__(name=model)

        assert model in ["Mixer-S/32", "Mixer-S/16", "Mixer-B/32", "Mixer-B/16", "Mixer-L/32", "Mixer-L/16", "Mixer-H/14"]

        if model == "Mixer-S/32":
            num_layers = 8
            patch_size = 32
            dim_feat = 512
            dim_expand_feat = 2048
            dim_expand_seq = 256
        elif model == "Mixer-S/16":
            num_layers = 8
            patch_size = 16
            dim_feat = 512
            dim_expand_feat = 2048
            dim_expand_seq = 256
        elif model == "Mixer-B/32":
            num_layers = 12
            patch_size = 32
            dim_feat = 768
            dim_expand_feat = 3072
            dim_expand_seq = 384
        elif model == "Mixer-B/16":
            num_layers = 12
            patch_size = 16
            dim_feat = 768
            dim_expand_feat = 3072
            dim_expand_seq = 384
        elif model == "Mixer-L/32":
            num_layers = 24
            patch_size = 32
            dim_feat = 1024
            dim_expand_feat = 4096
            dim_expand_seq = 512
        elif model == "Mixer-L/16":
            num_layers = 24
            patch_size = 16
            dim_feat = 1024
            dim_expand_feat = 4096
            dim_expand_seq = 512
        elif model == "Mixer-H/14":
            num_layers = 32
            patch_size = 14
            dim_feat = 1280
            dim_expand_feat = 5120
            dim_expand_seq = 640

        self.mixer = MLPMixerEncoder(
            num_layers=num_layers,
            patch_size=patch_size,
            input_height=input_shape[1],
            input_width=input_shape[2],
            input_channels=input_shape[0],
            dim_feat=dim_feat,
            dim_expand_feat=dim_expand_feat,
            dim_expand_seq=dim_expand_seq,
            act_fun="GELU",
            drop_rate=0.1
        )

        self.head = nn.Sequential(
            GlobalAvgPool1d(),
            Linear(in_features=dim_feat, out_features=dim_output)
        )

    def forward(self, x):

        x = self.mixer(x)

        x = self.head(x)

        return x

class ViC(Classifier):

    """ Vision Transformer (ViC-T, ViC-S, ViC-B, ViC-L, ViC-H)

    Models: 224 x 224
    ViC-T: 3.4 MParams  1.75 GFlops
    ViC-S: 29 MParams   12 GFlops
    ViC-B: 84 MParams   33 GFlops
    ViC-L: 315 MParams  109 GFlops
    ViC-H: 688 MParams  264 GFlops

    Models: 384 x 384
    ViC-S: 29 MParams   45 GFlops
    ViC-B: 84 MParams   118 GFlops
    ViC-L: 315 MParams  363 GFlops
    ViC-H: 688 MParams  978 GFlops
    
    """

    def __init__(self, input_shape=(3, 224, 224), dim_output=1000, model="ViC-S"):
        super(ViC, self).__init__(name=model)

        assert model in ["ViC-T", "ViC-S", "ViC-B", "ViC-L", "ViC-H"]

        if model == "ViC-T":
            dim_model = [16, 32, 64, 128, 256]
            num_heads = [1, 2, 2, 4, 4]
            num_blocks = [1, 2, 2, 2, 1]
            patch_size = [(4, 4), (2, 2), (1, 1), (1, 1), (1, 1)]
        elif model == "ViC-S":
            dim_model = [48, 96, 192, 384, 768]
            num_heads = [1, 2, 3, 6, 12]
            num_blocks = [1, 2, 2, 2, 1]
            patch_size = [(4, 4), (2, 2), (1, 1), (1, 1), (1, 1)]
        elif model == "ViC-B":
            dim_model = [64, 128, 256, 512, 1024]
            num_heads = [1, 2, 4, 8, 16]
            num_blocks = [2, 3, 3, 3, 2]
            patch_size = [(4, 4), (2, 2), (1, 1), (1, 1), (1, 1)]
        elif model == "ViC-L":
            dim_model = [96, 192, 384, 768, 1536]
            num_heads = [2, 3, 6, 12, 24]
            num_blocks = [4, 4, 4, 4, 4]
            patch_size = [(4, 4), (2, 2), (1, 1), (1, 1), (1, 1)]
        elif model == "ViC-H":
            dim_model = [128, 256, 512, 1024, 2048]
            num_heads = [2, 4, 8, 16, 32]
            num_blocks = [5, 5, 5, 5, 5]
            patch_size = [(2, 2), (2, 2), (1, 1), (1, 1), (1, 1)]

        self.stem = Conv2d(
            in_channels=input_shape[0],
            out_channels=dim_model[0],
            kernel_size=(3, 3),
            stride=(2, 2)
        )

        self.conformer = Conformer(
            dim_model=dim_model,
            mask=None,
            ff_ratio=4,
            att_params=[
                {"class":"PatchMultiHeadAttention", "num_heads": num_heads[0], "patch_size": patch_size[0]},
                {"class":"PatchMultiHeadAttention", "num_heads": num_heads[1], "patch_size": patch_size[1]},
                {"class":"PatchMultiHeadAttention", "num_heads": num_heads[2], "patch_size": patch_size[2]},
                {"class":"PatchMultiHeadAttention", "num_heads": num_heads[3], "patch_size": patch_size[3]},
                {"class":"PatchMultiHeadAttention", "num_heads": num_heads[4], "patch_size": patch_size[4]}],
            num_blocks=num_blocks,
            pos_embedding=None,
            max_pos_encoding=None,
            drop_rate=0.1,
            causal=False,
            conv_stride=2,
            att_stride=1,
            conv_params={"class": "Conv2d", "params": {"padding": "same", "kernel_size": 3}}
        )

        self.reduction = GlobalAvgPool2d(dim=(1, 2))

        self.head = Linear(
            in_features=dim_model[-1],
            out_features=dim_output,
        )

    def forward(self, x):

        # (B, C, H, W) -> (B, H//2, W//2, D1)
        x = self.stem(x).permute(0, 2, 3, 1)

        # (B, H//2, W//2, D1) -> (B, H//32, W//32, D5)
        x, _, _, _ = self.conformer(x)

        # (B, H//32, W//32, D5) -> (B, D5)
        x =  self.reduction(x)

        # (B, D5) -> (B, Dout)
        x = self.head(x)

        return x

class ViViT(Classifier):

    """ Video Vision Transformer (ViViT-B/16x2, ViViT-L/16x2, ViViT-H/16x2)

    Models: 16 x 224 x 224
    ViViT-B/16x2: 87,749,776 Params     360 GFlops
    ViViT-L/16x2: 305,902,992 Params    1193 GFlops
    ViViT-H/16x2: 634,170,000 Params    2381 GFlops

    Models: 16 x 384 x 384
    ViViT-B/16x2: 
    ViViT-L/16x2: 
    ViViT-H/16x2:
    
    Reference: "ViViT: A Video Vision Transformer" by Arnab et al.
    https://arxiv.org/abs/2103.15691
    
    """

    def __init__(self, input_shape=(3, 16, 224, 224), dim_output=400, model="ViViT-B/16x2"):
        super(ViViT, self).__init__(name=model)

        assert model in ["ViViT-B/16x2", "ViViT-L/16x2", "ViViT-H/16x2"]

        if model == "ViViT-B/16x2":
            dim_model = 768
            patch_size = (2, 16, 16)
            num_heads = 12
            num_blocks = 12
        elif model == "ViViT-L/16x2":
            dim_model = 1024
            patch_size = (2, 16, 16)
            num_heads = 16
            num_blocks = 24
        elif model == "ViViT-H/16x2":
            dim_model = 1280
            patch_size = (2, 16, 16)
            num_heads = 16
            num_blocks = 32

        self.patch_emb = Conv3d(
            in_channels=input_shape[0],
            out_channels=dim_model,
            kernel_size=patch_size,
            stride=patch_size
        )

        self.class_embedding = Embedding(num_embeddings=1, embedding_dim=dim_model)

        self.transformer = Transformer(
            dim_model=dim_model,
            mask=None,
            ff_ratio=4,
            att_params={"class":"MultiHeadAttention", "num_heads": num_heads},
            num_blocks=num_blocks,
            pos_embedding="Embedding",
            max_pos_encoding=(input_shape[1] * input_shape[2] * input_shape[3]) // (patch_size[0] * patch_size[1] * patch_size[2]) + 1,
            drop_rate=0.1,
            causal=False,
            inner_dropout=True
        )

        self.head = Linear(
            in_features=dim_model,
            out_features=dim_output,
        )

    def forward(self, x):

        # (B, C, T, H, W) -> (B, T//Pt, H//Ph, W//Pw, D)
        x = self.patch_emb(x).permute(0, 2, 3, 4, 1)

        # (B, T//Pt, H//Ph, W//Pw, D) -> (B, N, D)
        x = x.flatten(start_dim=1, end_dim=3)

        # Class Token (B, 1, D)
        class_token = self.class_embedding(x.new_zeros(x.size(0), dtype=torch.int)).unsqueeze(1)

        # (B, N, D) -> (B, 1 + N, D)
        x = torch.cat([class_token, x], dim=1)

        # Transformer
        x, _, _ = self.transformer(x)

        # (B, N, D) -> (B, Dout)
        x = self.head(x[:, 0])

        return x

class ViViC(Classifier):

    """ Vision Transformer (ViViC-T, ViViC-S, ViViC-B, ViViC-L, ViViC-H)

    Models: 16 x 224 x 224
    ViViC-T: 11,014,960 Params  34 GFlops
    ViViC-S: 
    ViViC-B: 281 GFlops
    ViViC-L: 
    ViViC-H: 

    Models: 16 x 384 x 384
    ViViC-T:
    ViViC-S: 
    ViViC-B: 
    ViViC-L: 
    ViViC-H: 
    
    """

    def __init__(self, input_shape=(3, 16, 224, 224), dim_output=400, model="ViViC-S"):
        super(ViViC, self).__init__(name=model)

        assert model in ["ViViC-T", "ViViC-S", "ViViC-B", "ViViC-L", "ViViC-H"]

        if model == "ViViC-T":
            dim_model = [32, 64, 128, 256, 512]
            num_heads = [1, 2, 4, 4, 8]
            num_blocks = [1, 1, 1, 1, 1]
            patch_size = [(1, 4, 4), (1, 4, 4), (1, 2, 2), (1, 1, 1), (1, 1, 1)]
        elif model == "ViViC-S":
            dim_model = [48, 96, 192, 384, 768]
            num_heads = [1, 2, 3, 6, 12]
            num_blocks = [1, 2, 2, 2, 1]
            patch_size = [(1, 4, 4), (1, 4, 4), (1, 2, 2), (1, 1, 1), (1, 1, 1)]
        elif model == "ViViC-B":
            dim_model = [64, 128, 256, 512, 1024]
            num_heads = [1, 2, 4, 8, 16]
            num_blocks = [2, 3, 3, 3, 2]
            patch_size = [(1, 4, 4), (1, 4, 4), (1, 2, 2), (1, 1, 1), (1, 1, 1)]
        elif model == "ViViC-L":
            dim_model = [96, 192, 384, 768, 1536]
            num_heads = [2, 3, 6, 12, 24]
            num_blocks = [4, 4, 4, 4, 4]
            patch_size = [(1, 4, 4), (1, 4, 4), (1, 2, 2), (1, 1, 1), (1, 1, 1)]
        elif model == "ViViC-H":
            dim_model = [128, 256, 512, 1024, 2048]
            num_heads = [2, 4, 8, 16, 32]
            num_blocks = [5, 5, 5, 5, 5]
            patch_size = [(1, 4, 4), (1, 4, 4), (1, 2, 2), (1, 1, 1), (1, 1, 1)]

        self.stem = Conv3d(
            in_channels=input_shape[0],
            out_channels=dim_model[0],
            kernel_size=(3, 3, 3),
            stride=(2, 2, 2)
        )

        self.conformer = Conformer(
            dim_model=dim_model,
            mask=None,
            ff_ratio=4,
            att_params=[
                {"class":"Patch3DMultiHeadAttention", "num_heads": num_heads[0], "patch_size": patch_size[0]},
                {"class":"Patch3DMultiHeadAttention", "num_heads": num_heads[1], "patch_size": patch_size[1]},
                {"class":"Patch3DMultiHeadAttention", "num_heads": num_heads[2], "patch_size": patch_size[2]},
                {"class":"Patch3DMultiHeadAttention", "num_heads": num_heads[3], "patch_size": patch_size[3]},
                {"class":"Patch3DMultiHeadAttention", "num_heads": num_heads[4], "patch_size": patch_size[4]}],
            num_blocks=num_blocks,
            pos_embedding=None,
            max_pos_encoding=None,
            drop_rate=0.1,
            causal=False,
            conv_stride=2,
            att_stride=1,
            conv_params={"class": "Conv3d", "params": {"padding": "same", "kernel_size": 3}}
        )

        self.reduction = GlobalAvgPool3d(axis=(1, 2, 3))

        self.head = Linear(
            in_features=dim_model[-1],
            out_features=dim_output,
        )

    def forward(self, x):

        # (B, C, T, H, W) -> (B, T//2, H//2, W//2, D1)
        x = self.stem(x).permute(0, 2, 3, 4, 1)

        # (B, T//2, H//2, W//2, D1) -> (B, T//2, H//32, W//32, D5)
        x = self.conformer(x)

        # (B, T//2, H//32, W//32, D5) -> (B, D5)
        x = self.reduction(x)

        # (B, D5) -> (B, Dout)
        x = self.head(x)

        return x