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

# Layers
from nnet.layers import (
    Embedding,
    Linear
)

###############################################################################
# Multi-Head Attention Layers
###############################################################################

class MultiHeadAttention(nn.Module):

    """Mutli-Head Attention Layer

    Args:
        dim_model: model feature dimension
        num_heads: number of attention heads

    References: 
        Attention Is All You Need, Vaswani et al.
        https://arxiv.org/abs/1706.03762

    """

    def __init__(self, dim_model, num_heads, **kargs):
        super(MultiHeadAttention, self).__init__()

        # Attention Params
        self.num_heads = num_heads # H
        self.dim_model = dim_model # D
        self.dim_head = dim_model // num_heads # d

        # Linear Layers
        self.query_layer = Linear(self.dim_model, self.dim_model)
        self.key_layer = Linear(self.dim_model, self.dim_model)
        self.value_layer = Linear(self.dim_model, self.dim_model)
        self.output_layer = Linear(self.dim_model, self.dim_model)

    def forward(self, Q, K, V, mask=None, **kargs):

        """Scaled Dot-Product Multi-Head Attention

        Args:
            Q: Query of shape (B, T, D)
            K: Key of shape (B, T, D)
            V: Value of shape (B, T, D)
            mask: Optional position mask of shape (1 or B, 1 or H, 1 or T, 1 or T)
        
        Return:
            O: Attention output of shape (B, T, D)
            att_w: Attention weights of shape (B, H, T, T)

        """

        # Batch size B
        batch_size = Q.size(0)

        # Linear Layers
        Q = self.query_layer(Q)
        K = self.key_layer(K)
        V = self.value_layer(V)

        # Reshape and Transpose (B, T, D) -> (B, H, T, d)
        Q = Q.reshape(batch_size, -1, self.num_heads, self.dim_head).transpose(1, 2)
        K = K.reshape(batch_size, -1, self.num_heads, self.dim_head).transpose(1, 2)
        V = V.reshape(batch_size, -1, self.num_heads, self.dim_head).transpose(1, 2)

        # Att scores (B, H, T, T)
        att_scores = Q.matmul(K.transpose(2, 3)) / K.shape[-1]**0.5

        # Apply mask
        if mask is not None:
            att_scores += (mask.logical_not() * -1e9)

        # Att weights (B, H, T, T)
        att_w = att_scores.softmax(dim=-1)

        # Att output (B, H, T, d)
        O = att_w.matmul(V)

        # Transpose and Reshape (B, H, T, d) -> (B, T, D)
        O = O.transpose(1, 2).reshape(batch_size, -1,  self.dim_model)

        # Output linear layer
        O = self.output_layer(O)

        return O, att_w.detach(), None

    def pad(self, Q, K, V, mask, chunk_size):

        # Compute Overflows
        overflow_Q = Q.size(1) % chunk_size
        overflow_KV = K.size(1) % chunk_size
        
        padding_Q = chunk_size - overflow_Q if overflow_Q else 0
        padding_KV = chunk_size - overflow_KV if overflow_KV else 0

        batch_size, seq_len_KV, _ = K.size()

        # Input Padding (B, T, D) -> (B, T + P, D)
        Q = F.pad(Q, (0, 0, 0, padding_Q), value=0)
        K = F.pad(K, (0, 0, 0, padding_KV), value=0)
        V = F.pad(V, (0, 0, 0, padding_KV), value=0)

        # Update Padding Mask
        if mask is not None:

            # (B, 1, 1, T) -> (B, 1, 1, T + P) 
            if mask.size(2) == 1:
                mask = F.pad(mask, pad=(0, padding_KV), value=1)
            # (B, 1, T, T) -> (B, 1, T + P, T + P)
            else:
                mask = F.pad(mask, pad=(0, padding_Q, 0, padding_KV), value=1)

        elif padding_KV:

            # None -> (B, 1, 1, T + P) 
            mask = F.pad(Q.new_zeros(batch_size, 1, 1, seq_len_KV), pad=(0, padding_KV), value=1)

        return Q, K, V, mask, padding_Q

class MultiDimensionalMultiHeadAttention(MultiHeadAttention):

    """ Multi Dimensional Multi Head Attention 

    Flatten dimensions before performing classical multi-head attention
    
    """

    def __init__(self, dim_model, num_heads, **kargs):
        super(MultiDimensionalMultiHeadAttention, self).__init__(dim_model, num_heads, **kargs)

    def forward(self, Q, K, V, mask=None, **kargs):

        # Shape
        shape = Q.shape

        # Flatten Tokens (B, ..., D) -> (B, N, D)
        Q = Q.flatten(start_dim=1, end_dim=-2)
        K = K.flatten(start_dim=1, end_dim=-2)
        V = V.flatten(start_dim=1, end_dim=-2)

        # Multi Head Attention
        O, att_w, _ = super(MultiDimensionalMultiHeadAttention, self).forward(Q=Q, K=K, V=V, mask=mask)

        # Unflatten (B, N, D) -> (B, ..., D)
        O = O.reshape(shape)

        return O, att_w, None

class PatchMultiHeadAttention(MultiDimensionalMultiHeadAttention):

    """Patch Mutli-Head Attention Layer

    Perform an average pooling on query, key and value linear projections before attention.
    Reduce attention complexity from O(N**2·D) to O(N**2.D/P**2) where N = H * W and
    P = Ph * Pw. Upsample output projection to input shape.

    Args:
        dim_model: model feature dimension
        num_heads: number of attention heads
        pool_size: attention pooling size (Ph, Pw)
    
    """

    def __init__(self, dim_model, num_heads, patch_size, **kargs):
        super(PatchMultiHeadAttention, self).__init__(dim_model, num_heads, **kargs)

        # Attention Params
        self.downsample = nn.AvgPool2d(kernel_size=patch_size, stride=patch_size)
        self.upsample = nn.Upsample(scale_factor=(patch_size, patch_size) if isinstance(patch_size, int) else tuple(patch_size))
        self.mask_pooling = nn.MaxPool2d(kernel_size=patch_size, stride=patch_size)

    def forward(self, Q, K, V, mask=None, **args):

        # AvgPool2d (B, H, W, D) -> (B, H/Ph, W/Pw, D)
        Q = self.downsample(Q.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        K = self.downsample(K.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        V = self.downsample(V.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        # Min Pool Mask
        # mask = - self.mask_pooling(- mask)

        # Self-Attention
        O, att_w, _ = super(PatchMultiHeadAttention, self).forward(Q=Q, K=K, V=V, mask=mask)

        # UpSample (B, H/Ph, W/Pw, D) -> (B, H, W, D)
        O = self.upsample(O.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        return O, att_w, None

class Patch3DMultiHeadAttention(MultiHeadAttention):

    """Patch 3D Mutli-Head Attention Layer

    Perform an average pooling on query, key and value linear projections before attention.
    Reduce attention complexity from O(N**2·D) to O(N**2.D/P**2) where N = T * H * W and
    P = Pt * Ph * Pw. Upsample output projection to input shape.

    Args:
        dim_model: model feature dimension
        num_heads: number of attention heads
        pool_size: attention pooling size (Pt, Ph, Pw)
    
    """

    def __init__(self, dim_model, num_heads, patch_size, **kargs):
        super(Patch3DMultiHeadAttention, self).__init__(dim_model, num_heads, **kargs)

        # Attention Params
        self.downsample = nn.AvgPool3d(kernel_size=patch_size, stride=patch_size)
        self.upsample = nn.Upsample(scale_factor=(patch_size, patch_size, patch_size) if isinstance(patch_size, int) else tuple(patch_size))
        self.mask_pooling = nn.MaxPool3d(kernel_size=patch_size, stride=patch_size)

    def forward(self, Q, K, V, mask=None, **args):

        # AvgPool3d (B, T, H, W, D) -> (B, T/Pt, H/Ph, W/Pw, D)
        Q = self.downsample(Q.permute(0, 4, 1, 2, 3)).permute(0, 2, 3, 4, 1)
        K = self.downsample(K.permute(0, 4, 1, 2, 3)).permute(0, 2, 3, 4, 1)
        V = self.downsample(V.permute(0, 4, 1, 2, 3)).permute(0, 2, 3, 4, 1)

        # Query shape (B, T/Pt, H/Ph, W/Pw, D)
        shape = Q.shape

        # Flatten Tokens (B, T/Pt, H/Ph, W/Pw, D) -> (B, N, D)
        Q = Q.flatten(start_dim=1, end_dim=3)
        K = K.flatten(start_dim=1, end_dim=3)
        V = V.flatten(start_dim=1, end_dim=3)

        # Min Pool Mask (B or 1, 1, T*H*W, T*H*W) -> (B or 1, 1, N, N)
        #print(mask, mask.shape)
        #mask = - self.mask_pooling(- mask.reshape(mask.size(0), ))
        #print(mask, mask.shape)
        #exit()

        # Self-Attention
        O, att_w, _ = super(Patch3DMultiHeadAttention, self).forward(Q=Q, K=K, V=V, mask=mask)

        # Unflatten (B, N, D) -> (B, T/Pt, H/Ph, W/Pw, D)
        O = O.reshape(shape)

        # UpSample (B, T/Pt, H/Ph, W/Pw, D) -> (B, T, H, W, D)
        O = self.upsample(O.permute(0, 4, 1, 2, 3)).permute(0, 2, 3, 4, 1)

        return O, att_w, None

class AxialMultiHeadAttention(MultiHeadAttention):

    """Axial Mutli-Head Attention Layer

    Axial Attention permorms attention over a single axis for multidimensional data (B, D1, D2, ..., Dd, D). 
    Assuming D1=D2=...=Dd, axial attention reduce attention complexity from O(N**2·D) to O(N.N**(1/d).D) 
    where N=D1*D2*...*Dd.

    Args:
        dim_model: model feature dimension
        num_heads: number of attention heads
        axis: attention axis

    References: 
        AXIAL ATTENTION IN MULTIDIMENSIONAL TRANSFORMERS, Ho et al.
        https://arxiv.org/abs/1912.12180

    """

    def __init__(self, dim_model, num_heads, axis):
        super(AxialMultiHeadAttention, self).__init__(dim_model, num_heads)

        # Attention Params
        self.axis = axis

    def forward(self, Q, K, V, mask=None):

        # Linear Layers
        Q = self.query_layer(Q)
        K = self.key_layer(K)
        V = self.value_layer(V)

        # Reshape and Transpose (B, ..., Daxis, ..., D) -> (B, ..., H, Daxis, d)
        Q = Q.transpose(self.axis, -2).reshape(Q.shape[:-1] + (self.num_heads, self.dim_head)).transpose(-3, -2)
        K = K.transpose(self.axis, -2).reshape(K.shape[:-1] + (self.num_heads, self.dim_head)).transpose(-3, -2)
        V = V.transpose(self.axis, -2).reshape(V.shape[:-1] + (self.num_heads, self.dim_head)).transpose(-3, -2)

        # Att scores (B, ..., H, Daxis, Daxis)
        att_scores = Q.matmul(K.transpose(-2, -1)) / K.shape[-1]**0.5

        # Apply mask
        if mask is not None:
            att_scores += (mask.logical_not() * -1e9)

        # Att weights (B, ..., H, Daxis, Daxis)
        att_w = att_scores.softmax(dim=-1)

        # Att output (B, ..., H, Daxis, d)
        O = att_w.matmul(V)

        # Transpose and Reshape (B, ..., H, Daxis, d) -> (B, ..., Daxis, ..., D)
        O = O.transpose(-3, -2).reshape(O.shape[:-2] + (self.dim_model,)).transpose(self.axis, -2)

        # Output linear layer
        O = self.output_layer(O)

        return O, att_w.detach()

class GroupedMultiHeadAttention(MultiHeadAttention):

    """Grouped Mutli-Head Attention Layer

    Grouped multi-head attention reduces attention complexity from O(T2·D) to O(T2·D/G) 
    by grouping neighbouring time elements along the feature dimension before applying 
    scaled dot-product attention. 

    Args:
        dim_model: model feature dimension
        num_heads: number of attention heads
        group_size: attention group size

    """

    def __init__(self, dim_model, num_heads, group_size):
        super(GroupedMultiHeadAttention, self).__init__(dim_model, num_heads)

        # Attention Params
        self.group_size = group_size # G
        self.dim_head = (self.group_size * dim_model) // self.num_heads # d

    def forward(self, Q, K, V, mask=None):

        # Batch size B
        batch_size = Q.size(0)

        # Linear Layers
        Q = self.query_layer(Q)
        K = self.key_layer(K)
        V = self.value_layer(V)

        # Chunk Padding
        Q, K, V, mask, padding = self.pad(Q, K, V, mask, chunk_size=self.group_size)

        # Reshape and Transpose (B, T, D) -> (B, H, T//G, d)
        Q = Q.reshape(batch_size, -1, self.num_heads, self.dim_head).transpose(1, 2)
        K = K.reshape(batch_size, -1, self.num_heads, self.dim_head).transpose(1, 2)
        V = V.reshape(batch_size, -1, self.num_heads, self.dim_head).transpose(1, 2)

        # Att scores (B, H, T//G, T//G)
        att_scores = Q.matmul(K.transpose(2, 3)) / K.shape[-1]**0.5

        # Apply mask
        if mask is not None:

            # Slice Mask (B, 1, T, T) -> (B, 1, T//G, T//G)
            mask = mask[:, :, ::self.group_size, ::self.group_size]

            # Apply mask
            att_scores += (mask.logical_not() * -1e9)

        # Att weights (B, H, T//G, T//G)
        att_w = att_scores.softmax(dim=-1)

        # Att output (B, H, T//G, d)
        O = att_w.matmul(V)

        # Transpose and Reshape (B, H, T//G, d) -> (B, T, D)
        O = O.transpose(1, 2).reshape(batch_size, -1,  self.dim_model)

        # Slice Padding
        O = O[:, :O.size(1) - padding]

        # Output linear layer
        O = self.output_layer(O)

        return O, att_w.detach()

class LocalMultiHeadAttention(MultiHeadAttention):

    """Local Multi-Head Attention Layer

    Local multi-head attention restricts the attended positions to a local neighborhood 
    around the query position. This is achieved by segmenting the hidden sequence into 
    non overlapping blocks of size K and performing scaled dot-product attention in 
    parallel for each of these blocks.

    Args:
        dim_model: model feature dimension
        num_heads: number of attention heads
        kernel_size: attention kernel size / window 

    References:
        Image Transformer, Parmar et al.
        https://arxiv.org/abs/1802.05751

    """

    def __init__(self, dim_model, num_heads, kernel_size):
        super(LocalMultiHeadAttention, self).__init__(dim_model, num_heads)

        # Attention Params
        self.kernel_size = kernel_size # K

    def forward(self, Q, K, V, mask=None):

        # Batch size B
        batch_size = Q.size(0)

        # Linear Layers
        Q = self.query_layer(Q)
        K = self.key_layer(K)
        V = self.value_layer(V)

        # Chunk Padding
        Q, K, V, mask, padding = self.pad(Q, K, V, mask, chunk_size=self.kernel_size)

        # Reshape and Transpose (B, T, D) -> (B, T//K, H, K, d)
        Q = Q.reshape(batch_size, -1, self.kernel_size, self.num_heads, self.dim_head).transpose(2, 3)
        K = K.reshape(batch_size, -1, self.kernel_size, self.num_heads, self.dim_head).transpose(2, 3)
        V = V.reshape(batch_size, -1, self.kernel_size, self.num_heads, self.dim_head).transpose(2, 3)

        # Att scores (B, T//K, H, K, K)
        att_scores = Q.matmul(K.transpose(3, 4)) / K.shape[-1]**0.5

        # Apply mask
        if mask is not None:

            # Slice mask (B, 1, T, T) -> (B, T//K, 1, K, K)
            masks = []
            for m in range(mask.size(-1) // self.kernel_size):
                masks.append(mask[:, :, m * self.kernel_size : (m + 1) * self.kernel_size, m * self.kernel_size : (m + 1) * self.kernel_size])
            mask = torch.stack(masks, dim=1)

            # Apply mask
            att_scores = att_scores.float() - mask.float() * 1e9

        # Att weights (B, T//K, H, K, K)
        att_w = att_scores.softmax(dim=-1)

        # Att output (B, T//K, H, K, d)
        O = att_w.matmul(V)

        # Transpose and Reshape (B, T//K, H, K, d) -> (B, T, D)
        O = O.transpose(2, 3).reshape(batch_size, -1, self.dim_model)

        # Slice Padding
        O = O[:, :O.size(1) - padding]

        # Output linear layer
        O = self.output_layer(O)

        return O, att_w.detach()

class StridedMultiHeadAttention(MultiHeadAttention):

    """Strided Mutli-Head Attention Layer

    Strided multi-head attention performs global sequence downsampling by striding 
    the attention query bedore aplying scaled dot-product attention. This results in 
    strided attention maps where query positions can attend to the entire sequence 
    context to perform downsampling.

    Args:
        dim_model: model feature dimension
        num_heads: number of attention heads
        stride: query stride

    """

    def __init__(self, dim_model, num_heads, stride):
        super(StridedMultiHeadAttention, self).__init__(dim_model, num_heads)

        # Attention Params
        self.stride = stride # S

    def forward(self, Q, K, V, mask=None):

        # Query Subsampling (B, T, D) -> (B, T//S, D)
        Q = Q[:, ::self.stride]

        # Mask Subsampling (B, 1, T, T) -> (B, 1, T//S, T)
        if mask is not None:
            mask = mask[:, :, ::self.stride]

        # Multi-Head Attention
        return super(StridedMultiHeadAttention, self).forward(Q, K, V, mask)

class StridedLocalMultiHeadAttention(MultiHeadAttention):

    """Strided Local Multi-Head Attention Layer

    Args:
        dim_model: model feature dimension
        num_heads: number of attention heads
        kernel_size: attention kernel size / window
        stride: query stride
        
    """

    def __init__(self, dim_model, num_heads, kernel_size, stride):
        super(StridedLocalMultiHeadAttention, self).__init__(dim_model, num_heads)

        # Assert
        assert kernel_size % stride == 0, "Attention kernel size has to be a multiple of attention stride"

        # Attention Params
        self.kernel_size = kernel_size # K
        self.stride = stride # S

    def forward(self, Q, K, V, mask=None):

        # Batch size B
        batch_size = Q.size(0)

        # Query Subsampling (B, T, D) -> (B, T//S, D)
        Q = Q[:, ::self.stride]

        # Linear Layers
        Q = self.query_layer(Q)
        K = self.key_layer(K)
        V = self.value_layer(V)

        # Chunk Padding
        Q, K, V, mask, padding = self.pad(Q, K, V, mask, chunk_size=self.kernel_size)

        # Reshape and Transpose (B, T//S, D) -> (B, T//K, H, K//S, d)
        Q = Q.reshape(batch_size, -1, self.kernel_size//self.stride, self.num_heads, self.dim_head).transpose(2, 3)
        # Reshape and Transpose (B, T, D) -> (B, T//K, H, K, d)
        K = K.reshape(batch_size, -1, self.kernel_size, self.num_heads, self.dim_head).transpose(2, 3)
        V = V.reshape(batch_size, -1, self.kernel_size, self.num_heads, self.dim_head).transpose(2, 3)

        # Att scores (B, T//K, H, K//S, K)
        att_scores = Q.matmul(K.transpose(3, 4)) / K.shape[-1]**0.5

        # Apply mask
        if mask is not None:

            # Slice mask (B, 1, T, T) -> (B, T//K, 1, K, K)
            masks = []
            for m in range(mask.size(-1) // self.kernel_size):
                masks.append(mask[:, :, m * self.kernel_size : (m + 1) * self.kernel_size, m * self.kernel_size : (m + 1) * self.kernel_size])
            mask = torch.stack(masks, dim=1)

            # Subsample mask (B, T//K, 1, K, K) -> (B, T//K, 1, K//S, K)
            mask = mask[:, :, :, ::self.stride]

            # Apply mask
            att_scores = att_scores.float() - mask.float() * 1e9

        # Att weights (B, T//K, H, K//S, K)
        att_w = att_scores.softmax(dim=-1)

        # Att output (B, T//K, H, K//S, d)
        O = att_w.matmul(V)

        # Transpose and Reshape (B, T//K, H, K//S, d) -> (B, T//S, D)
        O = O.transpose(2, 3).reshape(batch_size, -1, self.dim_model)

        # Slice Padding
        O = O[:, :(O.size(1) - padding - 1)//self.stride + 1]

        # Output linear layer
        O = self.output_layer(O)

        return O, att_w.detach()

class MultiHeadLinearAttention(MultiHeadAttention):

    """Multi-Head Linear Attention

    Args:
        dim_model: model feature dimension
        num_heads: number of attention heads

    References: 
        Efficient Attention: Attention with Linear Complexities, Shen et al.
        https://arxiv.org/abs/1812.01243

        Efficient conformer-based speech recognition with linear attention, Li et al.
        https://arxiv.org/abs/2104.06865

    """

    def __init__(self, dim_model, num_heads):
        super(MultiHeadLinearAttention, self).__init__(dim_model, num_heads)

    def forward(self, Q, K, V):

        # Batch size B
        batch_size = Q.size(0)

        # Linear Layers
        Q = self.query_layer(Q)
        K = self.key_layer(K)
        V = self.value_layer(V)

        # Reshape and Transpose (B, T, D) -> (B, N, T, d)
        Q = Q.reshape(batch_size, -1, self.num_heads, self.dim_head).transpose(1, 2)
        K = K.reshape(batch_size, -1, self.num_heads, self.dim_head).transpose(1, 2)
        V = V.reshape(batch_size, -1, self.num_heads, self.dim_head).transpose(1, 2)

        # Global Context Vector (B, N, d, d)
        KV = (K / K.shape[-1]**(1.0/4.0)).softmax(dim=-2).transpose(2, 3).matmul(V)

        # Attention Output (B, N, T, d)
        O = (Q / Q.shape[-1]**(1.0/4.0)).softmax(dim=-1).matmul(KV)

        # Transpose and Reshape (B, N, T, d) -> (B, T, D)
        O = O.transpose(1, 2).reshape(batch_size, -1,  self.dim_model)

        # Output linear layer
        O = self.output_layer(O)

        return O, KV.detach()

###############################################################################
# Multi-Head Self-Attention Layers with Relative Sinusoidal Poditional Encodings
###############################################################################

class RelPosMultiHeadSelfAttention(MultiHeadAttention):

    """Multi-Head Self-Attention Layer with Relative Sinusoidal Positional Encodings

    Args:
        dim_model: model feature dimension
        num_heads: number of attention heads
        causal: whether the attention is causal or unmasked
        max_pos_encoding: maximum relative distance between elements

    References: 
        Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context, Dai et al.
        https://arxiv.org/abs/1901.02860

    """

    def __init__(self, dim_model, num_heads, causal, max_pos_encoding):
        super(RelPosMultiHeadSelfAttention, self).__init__(dim_model, num_heads)

        # Position Embedding Layer
        self.pos_layer = Linear(self.dim_model, self.dim_model)
        self.causal = causal

        # Global content and positional bias
        self.u = nn.Parameter(torch.Tensor(self.dim_model)) # Content bias
        nn.init.zeros_(self.u)
        self.v = nn.Parameter(torch.Tensor(self.dim_model)) # Pos bias
        nn.init.zeros_(self.v)

        # Relative Sinusoidal Positional Encodings
        self.rel_pos_enc = RelativeSinusoidalPositionalEncoding(max_pos_encoding, self.dim_model, self.causal)

    def rel_to_abs(self, att_scores):

        """Relative to absolute position indexing

        Args:
            att_scores: absolute-by-relative indexed attention scores of shape 
            (B, H, T, Th + 2*T-1) for full context and (B, H, T, Th + T) for causal context

        Return:
            att_scores: absolute-by-absolute indexed attention scores of shape (B, H, T, Th + T)

        References: 
            causal context:
            Music Transformer, Huang et al.
            https://arxiv.org/abs/1809.04281
            
            full context:
            Attention Augmented Convolutional Networks, Bello et al.
            https://arxiv.org/abs/1904.09925

        """

        # Causal Context
        if self.causal:

            # Att Scores (B, H, T, Th + T)
            batch_size, num_heads, seq_length1, seq_length2 = att_scores.size()

            # Column Padding (B, H, T, 1 + Th + T)
            att_scores = F.pad(att_scores, pad=(1, 0), value=0)

            # Flatten (B, H, T + TTh + TT)
            att_scores = att_scores.reshape(batch_size, num_heads, -1)

            # Start Padding (B, H, Th + T + TTh + TT)
            att_scores = F.pad(att_scores, pad=(seq_length2 - seq_length1, 0), value=0)

            # Reshape (B, H, 1 + T, Th + T)
            att_scores = att_scores.reshape(batch_size, num_heads, 1 + seq_length1, seq_length2)

            # Slice (B, H, T, Th + T)
            att_scores = att_scores[:, :, 1:]

        # Full Context
        else:

            # Att Scores (B, H, T, Th + 2*T-1)
            batch_size, num_heads, seq_length1, seq_length2 = att_scores.size()

            # Column Padding (B, H, T, Th + 2*T)
            att_scores = F.pad(att_scores, pad=(0, 1), value=0)

            # Flatten (B, H, TTh + 2*TT)
            att_scores = att_scores.reshape(batch_size, num_heads, -1)

            # End Padding (B, H, TTh + 2*TT + Th + T - 1)
            att_scores = F.pad(att_scores, pad=(0, seq_length2 - seq_length1), value=0)

            # Reshape (B, H, T + 1, Th + 2*T-1)
            att_scores = att_scores.reshape(batch_size, num_heads, 1 + seq_length1, seq_length2)

            # Slice (B, H, T, Th + T)
            att_scores = att_scores[:, :, :seq_length1, seq_length1-1:]

        return att_scores

    def forward(self, Q, K, V, mask=None, hidden=None, output_dict=False, **kargs):

        """Scaled Dot-Product Self-Attention with relative sinusoidal position encodings

        Args:
            Q: Query of shape (B, T, D)
            K: Key of shape (B, T, D)
            V: Value of shape (B, T, D)
            mask: Optional position mask of shape (1 or B, 1 or H, 1 or T, 1 or T)
            hidden: Optional Key and Value hidden states for decoding
        
        Return:
            O: Attention output of shape (B, T, D)
            att_w: Attention weights of shape (B, H, T, Th + T)
            hidden: Key and value hidden states

        """

        # Batch size B
        batch_size = Q.size(0)

        # Linear Layers
        Q = self.query_layer(Q)
        K = self.key_layer(K)
        V = self.value_layer(V)

        # Hidden State Provided
        if hidden:
            K = torch.cat([hidden["K"], K], dim=1)
            V = torch.cat([hidden["V"], V], dim=1)

        # Update Hidden State
        hidden = {"K": K.detach(), "V": V.detach()}

        # Add Bias
        Qu = Q + self.u
        Qv = Q + self.v

        # Relative Positional Embeddings (B, Th + 2*T-1, D) / (B, Th + T, D)
        E = self.pos_layer(self.rel_pos_enc(batch_size, Q.size(1), K.size(1) - Q.size(1)))

        # Reshape and Transpose (B, T, D) -> (B, H, T, d)
        Qu = Qu.reshape(batch_size, -1, self.num_heads, self.dim_head).transpose(1, 2)
        Qv = Qv.reshape(batch_size, -1, self.num_heads, self.dim_head).transpose(1, 2)
        # Reshape and Transpose (B, Th + T, D) -> (B, H, Th + T, d)
        K = K.reshape(batch_size, -1, self.num_heads, self.dim_head).transpose(1, 2)
        V = V.reshape(batch_size, -1, self.num_heads, self.dim_head).transpose(1, 2)
        # Reshape and Transpose (B, Th + 2*T-1, D) -> (B, H, Th + 2*T-1, d) / (B, Th + T, D) -> (B, H, Th + T, d)
        E = E.reshape(batch_size, -1, self.num_heads, self.dim_head).transpose(1, 2)

        # att_scores (B, H, T, Th + T)
        att_scores_K = Qu.matmul(K.transpose(2, 3))
        att_scores_E = self.rel_to_abs(Qv.matmul(E.transpose(2, 3)))
        att_scores = (att_scores_K + att_scores_E) / K.shape[-1]**0.5

        # Apply mask
        if mask is not None:
            att_scores += (mask * -1e9)

        # Att weights (B, H, T, Th + T)
        att_w = att_scores.softmax(dim=-1)

        # Att output (B, H, T, d)
        O = att_w.matmul(V)

        # Transpose and Reshape (B, H, T, d) -> (B, T, D)
        O = O.transpose(1, 2).reshape(batch_size, -1,  self.dim_model)

        # Output linear layer
        O = self.output_layer(O)

        return {"samples": O, "att_maps": att_w.detach(), "hidden": hidden} if output_dict else (O, att_w.detach(), hidden)

class RelPosPatchMultiHeadAttention(MultiHeadAttention):

    """Relative Position Patch Mutli-Head Attention Layer

    Args:
        dim_model: model feature dimension
        num_heads: number of attention heads
        pool_size: attention pooling size (Ph, Pw)
        down_height: downsampled input height (H/Ph)
        down_width: downsampled input width (W/Pw)
    """

    def __init__(self, dim_model, num_heads, patch_size, down_height, down_width, **args):
        super(RelPosPatchMultiHeadAttention, self).__init__(dim_model, num_heads)

        # Attention Params
        self.downsample = nn.AvgPool2d(kernel_size=patch_size, stride=patch_size)
        self.upsample = nn.Upsample(scale_factor=(patch_size, patch_size) if isinstance(patch_size, int) else tuple(patch_size))
        self.down_height = down_height
        self.down_width = down_width

        # Global content and positional bias
        self.u = nn.Parameter(torch.Tensor(self.dim_model)) # Content bias
        nn.init.zeros_(self.u)
        self.v = nn.Parameter(torch.Tensor(self.dim_model)) # Pos bias
        nn.init.zeros_(self.v)

        # Relative Learned Positional Encodings
        self.rel_pos_enc = Embedding(num_embeddings=(2 * down_height - 1) * (2 * down_width - 1), embedding_dim=dim_model)

    def rel_to_abs(self, att_scores):

        """2D Relative to absolute position indexing

        Args:
            att_scores: absolute-by-relative indexed attention scores of shape 
            (B, h, N, N') where N = H/Ph * W/Pw and N' = (2H/Ph-1) * (2W/Pw-1)

        Return:
            att_scores: absolute-by-absolute indexed attention scores of shape (B, h, N, N)

        References: 
            CoAtNet: Marrying Convolution and Attention for All Data Sizes, Dai et al.
            https://arxiv.org/abs/2106.04803

        """

        # Att Scores (B, h, N, N')
        batch_size, num_heads, seq_length1, seq_length2 = att_scores.size()

        # Column Padding (B, h, N, N' + 1)
        att_scores = F.pad(att_scores, pad=(0, 1), value=0)

        # Reshape (B, h, H/Ph, W/Pw * (N' + 1))
        att_scores = att_scores.reshape(batch_size, num_heads, self.down_height, -1)

        # Column Padding (B, h, H/Ph, W/Pw * (N' + 1) + 1)
        att_scores = F.pad(att_scores, pad=(0, 1), value=0)

        # Flatten (B, h, H/Ph * (W/Pw * (N' + 1) + 1))
        att_scores = att_scores.reshape(batch_size, num_heads, -1)

        # End Padding (B, h, N' - N - H/Ph)
        att_scores = F.pad(att_scores, pad=(0, seq_length2 - seq_length1 - self.down_height), value=0)

        # Reshape (B, h, N + 1, 2H/Ph-1, 2W/Pw-1)
        att_scores = att_scores.reshape(batch_size, num_heads, seq_length1 + 1, 2 * self.down_height - 1, 2 * self.down_width - 1)

        # Slice (B, h, N, H/Ph, W/Pw)
        att_scores = att_scores[:, :, :seq_length1, -self.down_height:, -self.down_width:]

        # Reshape (B, h, N, N)
        att_scores = att_scores.reshape(batch_size, num_heads, seq_length1, seq_length1)

        return att_scores

    def forward(self, Q, K, V, mask=None, **kargs):

        # AvgPool2d (B, H, W, D) -> (B, H/Ph, W/Pw, D)
        Q = self.downsample(Q.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        K = self.downsample(K.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        V = self.downsample(V.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        # Query shape (B, H/Ph, W/Pw, D)
        shape = Q.shape

        # Flatten Tokens (B, H/Ph, W/Pw, D) -> (B, N, D)
        Q = Q.flatten(start_dim=1, end_dim=2)
        K = K.flatten(start_dim=1, end_dim=2)
        V = V.flatten(start_dim=1, end_dim=2)

        # Min Pool Mask
        mask = None

        ###

        # Batch size B
        batch_size = Q.size(0)

        # Linear Layers
        Q = self.query_layer(Q)
        K = self.key_layer(K)
        V = self.value_layer(V)

        # Add Bias
        Qu = Q + self.u
        Qv = Q + self.v

        # Relative Positional Embeddings (B, (2H/Ph-1) * (2W/Pw-1), D)
        E = self.rel_pos_enc(torch.arange((2 * shape[1] - 1) * (2 * shape[2] - 1), device=Q.device)).repeat(batch_size, 1, 1)

        # Reshape and Transpose (B, N, D) -> (B, h, N, d)
        Qu = Qu.reshape(batch_size, -1, self.num_heads, self.dim_head).transpose(1, 2)
        Qv = Qv.reshape(batch_size, -1, self.num_heads, self.dim_head).transpose(1, 2)
        K = K.reshape(batch_size, -1, self.num_heads, self.dim_head).transpose(1, 2)
        V = V.reshape(batch_size, -1, self.num_heads, self.dim_head).transpose(1, 2)
        # Reshape and Transpose (B, (2H/Ph-1) * (2W/Pw-1), D) -> (B, h, (2H/Ph-1) * (2W/Pw-1), d)
        E = E.reshape(batch_size, -1, self.num_heads, self.dim_head).transpose(1, 2)

        # att_scores (B, h, N, N)
        att_scores_K = Qu.matmul(K.transpose(2, 3))
        att_scores_E = self.rel_to_abs(Qv.matmul(E.transpose(2, 3)))
        att_scores = (att_scores_K + att_scores_E) / K.shape[-1]**0.5

        # Apply mask
        if mask is not None:
            att_scores += (mask * -1e9)

        # Att weights (B, h, N, N)
        att_w = att_scores.softmax(dim=-1)

        # Att output (B, h, N, d)
        O = att_w.matmul(V)

        # Transpose and Reshape (B, h, N, d) -> (B, N, D)
        O = O.transpose(1, 2).reshape(batch_size, -1,  self.dim_model)

        # Output linear layer
        O = self.output_layer(O)

        ###

        # Unflatten (B, N, D) -> (B, H/Ph, W/Pw, D)
        O = O.reshape(shape)

        # UpSample (B, H/Ph, W/Pw, D) -> (B, H, W, D)
        O = self.upsample(O.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        return O, att_w, None

class GroupedRelPosMultiHeadSelfAttention(RelPosMultiHeadSelfAttention):

    """Grouped Multi-Head Self-Attention Layer with Relative Sinusoidal Positional Encodings

    Args:
        dim_model: model feature dimension
        num_heads: number of attention heads
        causal: whether the attention is causal or unmasked
        max_pos_encoding: maximum relative distance between elements
        group_size: attention group size

    """

    def __init__(self, dim_model, num_heads, causal, max_pos_encoding, group_size):
        super(GroupedRelPosMultiHeadSelfAttention, self).__init__(dim_model, num_heads, causal, max_pos_encoding)

        # Attention Params
        self.group_size = group_size # G
        self.dim_head = (self.group_size * dim_model) // self.num_heads # d

        # Grouped Relative Sinusoidal Positional Encodings
        self.rel_pos_enc = GroupedRelativeSinusoidalPositionalEncoding(max_pos_encoding, self.dim_model, self.group_size, self.causal)

    def forward(self, Q, K, V, mask=None, hidden=None):

        # Batch size B
        batch_size = Q.size(0)

        # Linear Layers
        Q = self.query_layer(Q)
        K = self.key_layer(K)
        V = self.value_layer(V)

        # Hidden State Provided
        if hidden:
            Kh = torch.cat([hidden["K"], K], dim=1)
            Vh = torch.cat([hidden["V"], V], dim=1)
            K = torch.cat([hidden["K"][:, hidden["K"].size(1)%self.group_size:], K], dim=1)
            V = torch.cat([hidden["V"][:, hidden["V"].size(1)%self.group_size:], V], dim=1)

            # Update Hidden State
            hidden = {"K": Kh.detach(), "V": Vh.detach()}

        else:

            # Update Hidden State
            hidden = {"K": K.detach(), "V": V.detach()}

        # Chunk Padding
        Q, K, V, mask, padding = self.pad(Q, K, V, mask, chunk_size=self.group_size)

        # Add Bias
        Qu = Q + self.u
        Qv = Q + self.v

        # Relative Positional Embeddings (B, Th + 2*T-G, D) / (B, Th + T, D)
        E = self.pos_layer(self.rel_pos_enc(batch_size, Q.size(1), K.size(1) - Q.size(1)))

        # Reshape and Transpose (B, T, D) -> (B, H, T//G, d)
        Qu = Qu.reshape(batch_size, -1, self.num_heads, self.dim_head).transpose(1, 2)
        Qv = Qv.reshape(batch_size, -1, self.num_heads, self.dim_head).transpose(1, 2)
        # Reshape and Transpose (B, Th + T, D) -> (B, H, Th//G + T//G, d)
        K = K.reshape(batch_size, -1, self.num_heads, self.dim_head).transpose(1, 2)
        V = V.reshape(batch_size, -1, self.num_heads, self.dim_head).transpose(1, 2)
        # Reshape and Transpose (B, Th + 2*T-G, D) -> (B, H, Th//G + 2*T//G-1, d) / (B, Th + T, D) -> (B, H, Th//G + T//G, d)
        E = E.reshape(batch_size, -1, self.num_heads, self.dim_head).transpose(1, 2)

        # att_scores (B, H, T//G, Th//G + T//G)
        att_scores_K = Qu.matmul(K.transpose(2, 3))
        att_scores_E = self.rel_to_abs(Qv.matmul(E.transpose(2, 3)))
        att_scores = (att_scores_K + att_scores_E) / K.shape[-1]**0.5

        # Apply mask
        if mask is not None:

            # Slice Mask (B, 1, T, T) -> (B, 1, T//G, T//G)
            mask = mask[:, :, ::self.group_size, ::self.group_size]

            # Apply mask
            att_scores += (mask * -1e9)

        # Att weights (B, H, T//G, Th//G + T//G)
        att_w = att_scores.softmax(dim=-1)

        # Att output (B, H, T//G, d)
        O = att_w.matmul(V)

        # Transpose and Reshape (B, H, T//G, d) -> (B, T, D)
        O = O.transpose(1, 2).reshape(batch_size, -1,  self.dim_model)

        # Slice Padding
        O = O[:, :O.size(1) - padding]

        # Output linear layer
        O = self.output_layer(O)

        return O, att_w.detach(), hidden

class LocalRelPosMultiHeadSelfAttention(RelPosMultiHeadSelfAttention):

    """Local Multi-Head Self-Attention with Relative Sinusoidal Positional Encodings

    Args:
        dim_model: model feature dimension
        num_heads: number of attention heads
        causal: whether the attention is causal or unmasked
        kernel_size: attention kernel size / window 

    References: 
        Music Transformer, Huang et al.
        https://arxiv.org/abs/1809.04281
        
    """

    def __init__(self, dim_model, num_heads, causal, kernel_size):
        super(LocalRelPosMultiHeadSelfAttention, self).__init__(dim_model, num_heads, causal, kernel_size)

        # Attention Params
        self.kernel_size = kernel_size # K 

    def rel_to_abs(self, att_scores):

        """Relative to absolute position indexing

        Args:
            att_scores: absolute-by-relative indexed attention scores of shape 
            (B, N, T, 2 * K - 1) for full context and (B, H, T, K) for causal context

        Return:
            att_scores: absolute-by-absolute indexed attention scores of shape (B, T//K, H, K, K)

        References: 
            Causal context:
            Music Transformer, Huang et al.
            https://arxiv.org/abs/1809.04281
        """

        # Causal Context
        if self.causal:

            # Att Scores (B, H, T, K)
            batch_size, num_heads, seq_length1, seq_length2 = att_scores.size()

            # Reshape (B, T//K, H, K, K)
            att_scores = att_scores.reshape(batch_size, -1, self.num_heads, self.kernel_size, self.kernel_size)

            # Column Padding (B, T//K, H, K, 1 + K)
            att_scores = F.pad(att_scores, pad=(1, 0), value=0)

            # Reshape (B, T//K, H, 1 + K, K)
            att_scores = att_scores.reshape(batch_size, -1, self.num_heads, self.kernel_size + 1, self.kernel_size)

            # Slice (B, T//K, H, K, K)
            att_scores = att_scores[:, :, :, 1:]

        # Full Context
        else:

            # Att Scores (B, H, T, 2 * K - 1)
            batch_size, num_heads, seq_length1, seq_length2 = att_scores.size()

            # Reshape (B, T//K, H, K, 2 * K - 1)
            att_scores = att_scores.reshape(batch_size, -1, self.num_heads, self.kernel_size, seq_length2)

            # Column Padding (B, T//K, H, K, 2 * K)
            att_scores = F.pad(att_scores, pad=(0, 1), value=0)

            # Flatten (B, T//K, H, K * 2 * K)
            att_scores = att_scores.reshape(batch_size, -1, self.num_heads, 2 * self.kernel_size**2)

            # End Padding (B, T//K, H, K * 2 * K + K - 1)
            att_scores = F.pad(att_scores, pad=(0, self.kernel_size - 1), value=0)

            # Reshape (B, T//K, H, K + 1, 2 * K - 1)
            att_scores = att_scores.reshape(batch_size, -1, self.num_heads, self.kernel_size + 1, seq_length2)

            # Slice (B, T//K, H, K, K)
            att_scores = att_scores[:, :, :, :self.kernel_size, self.kernel_size - 1:]

        return att_scores

    def forward(self, Q, K, V, mask=None, hidden=None):

        # Batch size B
        batch_size = Q.size(0)

        # Linear Layers
        Q = self.query_layer(Q)
        K = self.key_layer(K)
        V = self.value_layer(V)

        # Chunk Padding
        Q, K, V, mask, padding = self.pad(Q, K, V, mask, chunk_size=self.kernel_size)

        # Add Bias
        Qu = Q + self.u
        Qv = Q + self.v

        # Relative Positional Embeddings (B, 2*K-1, D) / (B, K, D)
        E = self.pos_layer(self.rel_pos_enc(batch_size))

        # Reshape and Transpose (B, T, D) -> (B, H, T, d)
        Qv = Qv.reshape(batch_size, -1, self.num_heads, self.dim_head).transpose(1, 2)
        # Reshape and Transpose (B, T, D) -> (B, T//K, H, K, d)
        Qu = Qu.reshape(batch_size, -1, self.kernel_size, self.num_heads, self.dim_head).transpose(2, 3)
        K = K.reshape(batch_size, -1, self.kernel_size, self.num_heads, self.dim_head).transpose(2, 3)
        V = V.reshape(batch_size, -1, self.kernel_size, self.num_heads, self.dim_head).transpose(2, 3)
        # Reshape and Transpose (B, 2*K-1, D) -> (B, H, 2*K-1, d) / (B, K, D) -> (B, H, K, d)
        E = E.reshape(batch_size, -1, self.num_heads, self.dim_head).transpose(1, 2)

        # att_scores (B, T//K, H, K, K)
        att_scores_K = Qu.matmul(K.transpose(3, 4))
        att_scores_E = self.rel_to_abs(Qv.matmul(E.transpose(2, 3)))
        att_scores = (att_scores_K + att_scores_E) / K.shape[-1]**0.5

        # Mask scores
        if mask is not None:

            # Diagonal Mask (B, 1, T, T) -> (B, T//K, 1, K, K)
            masks = []
            for m in range(mask.size(-1) // self.kernel_size):
                masks.append(mask[:, :, m * self.kernel_size : (m + 1) * self.kernel_size, m * self.kernel_size : (m + 1) * self.kernel_size])
            mask = torch.stack(masks, dim=1)

            # Apply Mask
            att_scores = att_scores.float() - mask.float() * 1e9

        # Attention weights (B, T//K, H, K, K)
        att_w = att_scores.softmax(dim=-1)

        # Attention output (B, T//K, H, K, d)
        O = att_w.matmul(V)

        # Transpose and Reshape (B, T//K, H, K, d) -> (B, T, D)
        O = O.transpose(2, 3).reshape(batch_size, -1, self.dim_model)

        # Slice Padding
        O = O[:, :O.size(1) - padding]

        # Output linear layer
        O = self.output_layer(O)

        return O, att_w.detach(), hidden

class StridedRelPosMultiHeadSelfAttention(RelPosMultiHeadSelfAttention):

    """Strided Multi-Head Self-Attention with Relative Sinusoidal Positional Encodings

    Args:
        dim_model: model feature dimension
        num_heads: number of attention heads
        causal: whether the attention is causal or unmasked
        max_pos_encoding: maximum relative distance between elements
        stride: query stride
    """

    def __init__(self, dim_model, num_heads, causal, max_pos_encoding, stride):
        super(StridedRelPosMultiHeadSelfAttention, self).__init__(dim_model, num_heads, causal, max_pos_encoding)

        # Attention Params
        self.stride = stride # S

    def rel_to_abs(self, att_scores):

        """Relative to absolute position indexing

        Args:
            att_scores: absolute-by-relative indexed attention scores of shape 
            (B, H, T//S, Th + 2 * T - 1) for full context and (B, H, T//S, Th + T) for causal context

        Return:
            att_scores: absolute-by-absolute indexed attention scores of shape (B, H, T//S,Th + T)

        """

        # Causal Context
        if self.causal:

            # Att Scores (B, H, T // S, Th + T)
            batch_size, num_heads, seq_length1, seq_length2 = att_scores.size()

            # Column Padding (B, H, T // S, Th + T + S)
            att_scores = F.pad(att_scores, pad=(1, self.stride-1), value=0)

            # Flatten (B, H, TTh//S + TT//S + T)
            att_scores = att_scores.reshape(batch_size, num_heads, -1)

            # Start Padding (B, H, TTh//S + TT//S + T + Th)
            att_scores = F.pad(att_scores, pad=(seq_length2 - self.stride*seq_length1, 0), value=0)

            # Reshape (B, H, 1 + T // S, Th + T)
            att_scores = att_scores.reshape(batch_size, num_heads, seq_length1 + 1, seq_length2)

            # Slice (B, H, T // S, Th + T)
            att_scores = att_scores[:, :, 1:]
            
        # Full Context
        else:

            # Att Scores (B, H, T // S, Th + 2*T-1)
            batch_size, num_heads, seq_length1, seq_length2 = att_scores.size()

            # Column Padding (B, H, T // S, Th + 2*T-1 + S)
            att_scores = F.pad(att_scores, pad=(0, self.stride), value=0)

            # Flatten (B, H, TTh//S + 2*TT//S - T//S + T)
            att_scores = att_scores.reshape(batch_size, num_heads, -1)

            # End Padding (B, H, TTh//S + 2*TT//S - T//S + Th + 2T-1)
            att_scores = F.pad(att_scores, pad=(0, seq_length2 - seq_length1 * self.stride), value=0)

            # Reshape (B, H, T//S + 1, Th + 2*T-1)
            att_scores = att_scores.reshape(batch_size, num_heads, seq_length1 + 1, seq_length2)

            # Slice (B, H, T // S, Th + T)
            att_scores = att_scores[:, :, :seq_length1, seq_length1 * self.stride - 1:]

        return att_scores

    def forward(self, Q, K, V, mask=None, hidden=None):

        # Batch size B
        batch_size = Q.size(0)

        # Linear Layers
        Q = self.query_layer(Q)
        K = self.key_layer(K)
        V = self.value_layer(V)

        # Hidden State Provided
        if hidden:
            K = torch.cat([hidden["K"], K], dim=1)
            V = torch.cat([hidden["V"], V], dim=1)

        # Update Hidden State
        hidden = {"K": K.detach(), "V": V.detach()}

        # Chunk Padding
        Q, K, V, mask, _ = self.pad(Q, K, V, mask, chunk_size=self.stride)

        # Query Subsampling (B, T, D) -> (B, T//S, D)
        Q = Q[:, ::self.stride]

        # Add Bias
        Qu = Q + self.u
        Qv = Q + self.v

        # Relative Positional Embeddings (B, Th + 2*T-1, D) / (B, Th + T, D)
        E = self.pos_layer(self.rel_pos_enc(batch_size, self.stride * Q.size(1), K.size(1) - self.stride * Q.size(1)))

        # Reshape and Transpose (B, T//S, D) -> (B, H, T//S, d)
        Qu = Qu.reshape(batch_size, -1, self.num_heads, self.dim_head).transpose(1, 2)
        Qv = Qv.reshape(batch_size, -1, self.num_heads, self.dim_head).transpose(1, 2)
        # Reshape and Transpose (B, Th + T, D) -> (B, H, Th + T, d)
        K = K.reshape(batch_size, -1, self.num_heads, self.dim_head).transpose(1, 2)
        V = V.reshape(batch_size, -1, self.num_heads, self.dim_head).transpose(1, 2)
        # Reshape and Transpose (B, Th + 2*T-1, D) -> (B, H, Th + 2*T-1, d) / (B, Th + T, D) -> (B, H, Th + T, d)
        E = E.reshape(batch_size, -1, self.num_heads, self.dim_head).transpose(1, 2)

        # att_scores (B, H, T//S, Th + T)
        att_scores_K = Qu.matmul(K.transpose(2, 3))
        att_scores_E = self.rel_to_abs(Qv.matmul(E.transpose(2, 3)))
        att_scores = (att_scores_K + att_scores_E) / K.shape[-1]**0.5

        # Apply mask
        if mask is not None:

            # Mask Subsampling (B, 1, T, T) -> (B, 1, T//S, T)
            if mask is not None:
                mask = mask[:, :, ::self.stride]

            # Apply mask
            att_scores += (mask * -1e9)

        # Att weights (B, H, T//S, Th + T)
        att_w = att_scores.softmax(dim=-1)

        # Att output (B, H, T//S, d)
        O = att_w.matmul(V)

        # Transpose and Reshape (B, H, T//S, d) -> (B, T//S, D)
        O = O.transpose(1, 2).reshape(batch_size, -1,  self.dim_model)

        # Output linear layer
        O = self.output_layer(O)

        return O, att_w.detach(), hidden


class StridedLocalRelPosMultiHeadSelfAttention(RelPosMultiHeadSelfAttention):

    """Strided Local Multi-Head Self-Attention with Relative Sinusoidal Positional Encodings

    Args:
        dim_model: model feature dimension
        num_heads: number of attention heads
        causal: whether the attention is causal or unmasked
        kernel_size: attention kernel size / window 
        stride: query stride
    """

    def __init__(self, dim_model, num_heads, causal, kernel_size, stride):
        super(StridedLocalRelPosMultiHeadSelfAttention, self).__init__(dim_model, num_heads, causal, kernel_size)

        # Assert
        assert kernel_size % stride == 0, "Attention kernel size has to be a multiple of attention stride"

        # Attention Params
        self.kernel_size = kernel_size # K
        self.stride = stride # S

    def rel_to_abs(self, att_scores):

        """Relative to absolute position indexing

        Args:
            att_scores: absolute-by-relative indexed attention scores of shape 
            (B, H, T//S, 2 * K - 1) for full context and (B, H, T//S, K) for causal context

        Return:
            att_scores: absolute-by-absolute indexed attention scores of shape (B, T//K, H, K//S, K)
        """

        # Causal Context
        if self.causal:

            # Att Scores (B, H, T//S, K)
            batch_size, num_heads, seq_length1, seq_length2 = att_scores.size()

            # Reshape (B, T//K, H, K//S, K)
            att_scores = att_scores.reshape(batch_size, -1, self.num_heads, self.kernel_size//self.stride, self.kernel_size)

            # Column Padding (B, T//K, H, K//S, K + S)
            att_scores = F.pad(att_scores, pad=(1, self.stride - 1), value=0)

            # Reshape (B, T//K, H, 1 + K//S, K)
            att_scores = att_scores.reshape(batch_size, -1, self.num_heads, self.kernel_size//self.stride + 1, self.kernel_size)

            # Slice (B, T//K, H, K//S, K)
            att_scores = att_scores[:, :, :, 1:]

        # Full Context
        else:

            # Att Scores (B, H, T//S, 2*K-1)
            batch_size, num_heads, seq_length1, seq_length2 = att_scores.size()

            # Reshape (B, T//K, H, K//S, 2*K-1)
            att_scores = att_scores.reshape(batch_size, -1, self.num_heads, self.kernel_size//self.stride, seq_length2)

            # Column Padding (B, T//K, H, K//S, 2*K-1 + S)
            att_scores = F.pad(att_scores, pad=(0, self.stride), value=0)

            # Flatten (B, T//K, H, 2KK//S - K//S + K)
            att_scores = att_scores.reshape(batch_size, -1, self.num_heads, self.kernel_size//self.stride * (2 * self.kernel_size - 1 + self.stride))

            # End Padding (B, T//K, H, 2KK//S - K//S + 2K-1)
            att_scores = F.pad(att_scores, pad=(0, self.kernel_size - 1), value=0)

            # Reshape (B, T//K, H, K//S + 1, 2*K-1)
            att_scores = att_scores.reshape(batch_size, -1, self.num_heads, self.kernel_size//self.stride + 1, seq_length2)

            # Slice (B, T//K, H, K//S, K)
            att_scores = att_scores[:, :, :, :self.kernel_size//self.stride, self.kernel_size - 1:]

        return att_scores

    def forward(self, Q, K, V, mask=None, hidden=None):

        # Batch size B
        batch_size = Q.size(0)

        # Chunk Padding
        Q, K, V, mask, padding = self.pad(Q, K, V, mask, chunk_size=self.kernel_size)

        # Query Subsampling (B, T, D) -> (B, T//S, D)
        Q = Q[:, ::self.stride]

        # Linear Layers
        Q = self.query_layer(Q)
        K = self.key_layer(K)
        V = self.value_layer(V)

        # Add Bias
        Qu = Q + self.u
        Qv = Q + self.v

        # Relative Positional Embeddings (B, 2*K-1, D) / (B, K, D)
        E = self.pos_layer(self.rel_pos_enc(batch_size))

        # Reshape and Transpose (B, T//S, D) -> (B, H, T//S, d)
        Qv = Qu.reshape(batch_size, -1, self.num_heads, self.dim_head).transpose(1, 2)
        # Reshape and Transpose (B, T//S, D) -> (B, T//K, H, K//S, d)
        Qu = Qv.reshape(batch_size, -1, self.kernel_size//self.stride, self.num_heads, self.dim_head).transpose(2, 3)
        # Reshape and Transpose (B, T, D) -> (B, T//K, H, K, d)
        K = K.reshape(batch_size, -1, self.kernel_size, self.num_heads, self.dim_head).transpose(2, 3)
        V = V.reshape(batch_size, -1, self.kernel_size, self.num_heads, self.dim_head).transpose(2, 3)
        # Reshape and Transpose (B, 2*K-1, D) -> (B, H, 2*K-1, d) / (B, K, D) -> (B, H, K, d)
        E = E.reshape(batch_size, -1, self.num_heads, self.dim_head).transpose(1, 2)

        # att_scores (B, T//K, H, K//S, K)
        att_scores_K = Qu.matmul(K.transpose(3, 4))
        att_scores_E = self.rel_to_abs(Qv.matmul(E.transpose(2, 3)))
        att_scores = (att_scores_K + att_scores_E) / K.shape[-1]**0.5

        # Mask scores
        if mask is not None:

            # Diagonal Mask (B, 1, T, T) -> (B, T//K, 1, K, K)
            masks = []
            for m in range(mask.size(-1) // self.kernel_size):
                masks.append(mask[:, :, m * self.kernel_size : (m + 1) * self.kernel_size, m * self.kernel_size : (m + 1) * self.kernel_size])
            mask = torch.stack(masks, dim=1)

            # Stride Mask (B, T//K, 1, K, K) -> (B, T//K, 1, K//S, K)
            mask = mask[:, :, :, ::self.stride]

            # Apply Mask
            att_scores = att_scores.float() - mask.float() * 1e9

        # Attention weights (B, T//K, H, K//S, K)
        att_w = att_scores.softmax(dim=-1)

        # Attention output (B, T//K, H, K//S, d)
        O = att_w.matmul(V)

        # Transpose and Reshape (B, T//K, H, K//S, d) -> (B, T//S, D)
        O = O.transpose(2, 3).reshape(batch_size, -1, self.dim_model)

        # Slice Padding
        O = O[:, :(self.stride*O.size(1) - padding - 1)//self.stride + 1]

        # Output linear layer
        O = self.output_layer(O)

        return O, att_w.detach(), hidden

###############################################################################
# Positional Encodings
###############################################################################

class SinusoidalPositionalEncoding(nn.Module):
    
    """

    Sinusoidal Positional Encoding

    Reference: "Attention Is All You Need" by Vaswani et al.
    https://arxiv.org/abs/1706.03762

    """

    def __init__(self, max_len, dim_model):
        super(SinusoidalPositionalEncoding, self).__init__()

        pos_encoding = torch.zeros(max_len, dim_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        i = torch.arange(0, dim_model // 2, dtype=torch.float).unsqueeze(0) 
        angles = pos / 10000**(2 * i / dim_model)

        pos_encoding[:, 0::2] = angles.sin()
        pos_encoding[:, 1::2] = angles.cos()
        pos_encoding = pos_encoding.unsqueeze(0)

        self.register_buffer('pos_encoding', pos_encoding, persistent=False)

    def forward(self, batch_size=1, seq_len=None):

        # (B, T, D)
        if seq_len is not None:
            P = self.pos_encoding[:, :seq_len]

        # (B, Tmax, D)
        else:
            P = self.pos_encoding

        return P.repeat(batch_size, 1, 1)

class RelativeSinusoidalPositionalEncoding(nn.Module):
    
    """
        Relative Sinusoidal Positional Encoding

        Positional encoding for left context (sin) and right context (cos)
        Total context = 2 * max_len - 1
    """

    def __init__(self, max_len, dim_model, causal=False):
        super(RelativeSinusoidalPositionalEncoding, self).__init__()

        # PE
        pos_encoding = torch.zeros(2 * max_len - 1, dim_model)

        # Positions (max_len - 1, ..., max_len - 1)
        pos_left = torch.arange(start=max_len-1, end=0, step=-1, dtype=torch.float)
        pos_right = torch.arange(start=0, end=-max_len, step=-1, dtype=torch.float)
        pos = torch.cat([pos_left, pos_right], dim=0).unsqueeze(1)

        # Angles
        angles = pos / 10000**(2 * torch.arange(0, dim_model // 2, dtype=torch.float).unsqueeze(0) / dim_model)

        # Rel Sinusoidal PE
        pos_encoding[:, 0::2] = angles.sin()
        pos_encoding[:, 1::2] = angles.cos()

        pos_encoding = pos_encoding.unsqueeze(0)

        self.register_buffer('pos_encoding', pos_encoding, persistent=False)
        self.max_len = max_len
        self.causal = causal

    def forward(self, batch_size=1, seq_len=None, hidden_len=0):

        # Causal Context
        if self.causal:

            # (B, Th + T, D)
            if seq_len is not None:
                R = self.pos_encoding[:, self.max_len - seq_len - hidden_len : self.max_len]

            # (B, Tmax, D)
            else:
                R = self.pos_encoding[:,:self.max_len]

        # Full Context
        else:

            # (B, Th + 2*T-1, D)
            if seq_len is not None:
                R = self.pos_encoding[:, self.max_len - seq_len - hidden_len : self.max_len - 1  + seq_len]
            
            # (B, 2*Tmax-1, D)
            else:
                R = self.pos_encoding

        return R.repeat(batch_size, 1, 1)

class GroupedRelativeSinusoidalPositionalEncoding(nn.Module):
    
    """
        Relative Sinusoidal Positional Encoding for grouped multi-head attention

        Positional encoding for left context (sin) and right context (cos)
        Total context = 2 * max_len - group_size
    """

    def __init__(self, max_len, dim_model, group_size=1, causal=False):
        super(GroupedRelativeSinusoidalPositionalEncoding, self).__init__()

        # PE
        pos_encoding = torch.zeros(2 * max_len - group_size % 2, dim_model)

        # Positions (max_len - 1, ..., max_len - 1)
        pos_left = torch.arange(start=max_len-1, end=group_size % 2 - 1, step=-1, dtype=torch.float)
        pos_right = torch.arange(start=0, end=-max_len, step=-1, dtype=torch.float)
        pos = torch.cat([pos_left, pos_right], dim=0).unsqueeze(1)

        # Angles
        angles = pos / 10000**(2 * torch.arange(0, dim_model // 2, dtype=torch.float).unsqueeze(0) / dim_model)

        # Rel Sinusoidal PE
        pos_encoding[:, 0::2] = angles.sin()
        pos_encoding[:, 1::2] = angles.cos()

        pos_encoding = pos_encoding.unsqueeze(0)

        self.register_buffer('pos_encoding', pos_encoding, persistent=False)
        self.max_len = max_len
        self.causal = causal
        self.group_size = group_size

    def forward(self, batch_size=1, seq_len=None, hidden_len=0):

        # Causal Context
        if self.causal:

            # (B, Th + T, D)
            if seq_len is not None:
                R = self.pos_encoding[:, self.max_len - seq_len - hidden_len : self.max_len]
            
            # (B, Tmax, D)
            else:
                R = self.pos_encoding[:,:self.max_len]
        else:

            # (B, Th + 2*T-G, D)
            if seq_len is not None:
                R = self.pos_encoding[:, self.max_len - seq_len + self.group_size // 2 - hidden_len : self.max_len - self.group_size % 2  + seq_len - self.group_size // 2 ]
            
            # (B, 2*Tmax-G, D)
            else:
                R = self.pos_encoding

        return R.repeat(batch_size, 1, 1)

###############################################################################
# Attention Masks
###############################################################################

class PaddingMask(nn.Module):

    def __init__(self):
        super(PaddingMask, self).__init__()

    def forward(self, seq_len, x_len):

        # Init Mask (B, T)
        mask = x_len.new_zeros(x_len.size(0), seq_len)

        # Batch Loop
        for b in range(x_len.size(0)):
            mask[b, :x_len[b]] = x_len.new_ones(x_len[b])

        # Padding Mask (B, 1, 1, T)
        return mask[:, None, None, :]

class CausalTemporalMask3D(nn.Module):

    """ Causal Temporal Mask 3D: Causal Temporal Mask for spatiotemporal data """

    def __init__(self):
        super(CausalTemporalMask3D, self).__init__()
        self.padding_mask = PaddingMask()

    def forward(self, x, x_len=None):

        """ Forward

        Args:
            x: spatiotemporal data (B, T, H, W, D)
            x_len: data temporal length for post padding mask

        Return:
            spatiotemporal mask (1 or B, 1, N, N) with N = T * H * W
        
        """

        # Causal Mask (T, T)
        causal_mask = 1 - x.new_ones(x.size(1), x.size(1)).triu(1)

        # Spacial Upsampling (1, 1, N, N)
        causal_mask = F.interpolate(causal_mask[None, None], scale_factor=(x.size(2) * x.size(3), x.size(2) * x.size(3)))

        # Padding Mask
        if x_len is not None:

            # Padding Mask (B, 1, 1, T)
            padding_mask = self.padding_mask(x.size(1), x_len)

            # Spacial Upsampling (B, 1, 1, N)
            padding_mask = F.interpolate(padding_mask.type(causal_mask.dtype), scale_factor=(1, x.size(2) * x.size(3)))

            # Causal Mask Union Padding Mask (B, 1, N, N)
            return causal_mask.minimum(padding_mask)
        
        else:

            # Streaming Mask (1, 1, N, N)
            return causal_mask

class Mask(nn.Module):

    def __init__(self, left_context=None, right_context=None, seq_len_axis=1):
        super(Mask, self).__init__()
        self.padding_mask = PaddingMask()
        self.left_context = left_context
        self.right_context = right_context
        self.seq_len_axis = [seq_len_axis] if isinstance(seq_len_axis, int) else seq_len_axis

    def forward(self, x, x_len=None):

        # Seq Length T
        seq_len = torch.prod(torch.tensor([x.size(axis) for axis in self.seq_len_axis]))

        # Right Context Mask (T, T)
        if self.right_context != None:
            right_context_mask = 1 - x.new_ones(seq_len, seq_len).triu(diagonal=1+self.right_context)
        else:
            right_context_mask = 1 - x.new_ones(seq_len, seq_len).triu(diagonal=seq_len)

        # Left Context Mask (T, T)
        if self.left_context != None:
            left_context_mask = x.new_ones(seq_len, seq_len).triu(diagonal=-self.left_context)
        else:
            left_context_mask = x.new_ones(seq_len, seq_len).triu(diagonal=1-seq_len)

        # Context Mask (T, T)
        context_mask = right_context_mask.minimum(left_context_mask)

        # Padding Mask
        if x_len is not None:

            # Padding Mask (B, 1, 1, T)
            padding_mask = self.padding_mask(seq_len, x_len)

            # Context Mask Union Padding Mask (B, 1, T, T)
            return context_mask.minimum(padding_mask)
        
        else:

            # Streaming Mask (1, 1, T, T)
            return context_mask[None, None, :, :]

###############################################################################
# Attention Dictionary
###############################################################################

att_dict = {

    # Abs Attentions
    "MultiHeadAttention": MultiHeadAttention,
    "MultiDimensionalMultiHeadAttention": MultiDimensionalMultiHeadAttention,
    "PatchMultiHeadAttention": PatchMultiHeadAttention,
    "Patch3DMultiHeadAttention": Patch3DMultiHeadAttention,
    "AxialMultiHeadAttention": AxialMultiHeadAttention,
    "GroupedMultiHeadAttention": GroupedMultiHeadAttention,
    "LocalMultiHeadAttention": LocalMultiHeadAttention,
    "StridedMultiHeadAttention": StridedMultiHeadAttention,
    "StridedLocalMultiHeadAttention": StridedLocalMultiHeadAttention,
    "MultiHeadLinearAttention": MultiHeadLinearAttention,
    
    # Rel Attentions
    "RelPosMultiHeadSelfAttention": RelPosMultiHeadSelfAttention,
    "RelPosPatchMultiHeadAttention": RelPosPatchMultiHeadAttention,
    "GroupedRelPosMultiHeadSelfAttention": GroupedRelPosMultiHeadSelfAttention,
    "LocalRelPosMultiHeadSelfAttention": LocalRelPosMultiHeadSelfAttention,
    "StridedRelPosMultiHeadSelfAttention": StridedRelPosMultiHeadSelfAttention,
    "StridedLocalRelPosMultiHeadSelfAttention": StridedLocalRelPosMultiHeadSelfAttention
}