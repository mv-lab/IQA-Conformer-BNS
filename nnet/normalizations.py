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

# modules
from nnet.layers import (
    Permute,
    Transpose
)

###############################################################################
# Normalization Layers
###############################################################################

class LayerNorm(nn.LayerNorm):

    def __init__(self, normalized_shape, eps=1e-05, elementwise_affine=True, device=None, dtype=None, channels_last=True):
        super(LayerNorm, self).__init__(normalized_shape=normalized_shape, eps=eps, elementwise_affine=elementwise_affine, device=device, dtype=dtype)

        # Channels Last
        if channels_last:
            self.transpose = nn.Identity()
        else:
            self.transpose = Transpose(dim0=1, dim1=-1)

    def forward(self, input):

        return self.transpose(super(LayerNorm, self).forward(self.transpose(input)))

class BatchNorm1d(nn.BatchNorm1d):

    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None, channels_last=False):
        super(BatchNorm1d, self).__init__(num_features=num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats, device=device, dtype=dtype)

        # Channels Last
        if channels_last:
            self.input_permute = Permute(dims=(0, 2, 1))
            self.output_permute = Permute(dims=(0, 2, 1))
        else:
            self.input_permute = nn.Identity()
            self.output_permute = nn.Identity()

    def forward(self, input):

        return self.output_permute(super(BatchNorm1d, self).forward(self.input_permute(input)))

class BatchNorm2d(nn.BatchNorm2d):

    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None, channels_last=False):
        super(BatchNorm2d, self).__init__(num_features=num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats, device=device, dtype=dtype)

        # Channels Last
        if channels_last:
            self.input_permute = Permute(dims=(0, 3, 1, 2))
            self.output_permute = Permute(dims=(0, 2, 3, 1))
        else:
            self.input_permute = nn.Identity()
            self.output_permute = nn.Identity()

    def forward(self, input):

        return self.output_permute(super(BatchNorm2d, self).forward(self.input_permute(input)))

class BatchNorm3d(nn.BatchNorm3d):

    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None, channels_last=False):
        super(BatchNorm3d, self).__init__(num_features=num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats, device=device, dtype=dtype)

        # Channels Last
        if channels_last:
            self.input_permute = Permute(dims=(0, 4, 1, 2, 3))
            self.output_permute = Permute(dims=(0, 2, 3, 4, 1))
        else:
            self.input_permute = nn.Identity()
            self.output_permute = nn.Identity()

    def forward(self, input):

        return self.output_permute(super(BatchNorm3d, self).forward(self.input_permute(input)))

class InstanceNorm2d(nn.InstanceNorm2d):

    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False, device=None, dtype=None, channels_last=False):
        super(InstanceNorm2d, self).__init__(num_features=num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats, device=device, dtype=dtype)

        # Channels Last
        if channels_last:
            self.input_permute = Permute(dims=(0, 3, 1, 2))
            self.output_permute = Permute(dims=(0, 2, 3, 1))
        else:
            self.input_permute = nn.Identity()
            self.output_permute = nn.Identity()

    def forward(self, input):

        return self.output_permute(super(InstanceNorm2d, self).forward(self.input_permute(input)))

###############################################################################
# Normalization Dictionary
###############################################################################

norm_dict = {
    None: nn.Identity,
    "LayerNorm": LayerNorm,
    "BatchNorm1d": BatchNorm1d,
    "BatchNorm2d": BatchNorm2d,
    "BatchNorm3d": BatchNorm3d,
    "InstanceNorm2d": InstanceNorm2d,
    "LayerNorm": nn.LayerNorm
}