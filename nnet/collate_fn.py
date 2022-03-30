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

###############################################################################
# Collate Functions
###############################################################################

class CollateList(nn.Module):

    def __init__(self, inputs_axis=[0], targets_axis=[1]):
        super(CollateList, self).__init__()
        self.inputs_axis = inputs_axis
        self.targets_axis = targets_axis

    def forward(self, samples):

        inputs = []
        for axis in self.inputs_axis:
            inputs.append(torch.stack([sample[axis] for sample in samples], axis=0))
        inputs = inputs[0] if len(inputs) == 1 else inputs

        targets = []
        for axis in self.targets_axis:
            targets.append(torch.stack([sample[axis] for sample in samples], axis=0))
        targets = targets[0] if len(targets) == 1 else targets

        return {"inputs": inputs, "targets": targets}
