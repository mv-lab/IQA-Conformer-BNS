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
import torchvision
import torch.nn as nn
import torch.nn.functional as F

# Base Model
from nnet.model import Model

# Losses
from nnet.losses import (
    MeanSquaredError,
    SoftmaxCrossEntropy
)

# Metrics
from nnet.metrics import (
    CategoricalAccuracy,
)

# Decoders
from nnet.decoders import (
    ArgMaxDecoder,
)

# Collate Functions
from nnet.collate_fn import (
    CollateList,
)


###############################################################################
# Models
###############################################################################

class Classifier(Model):

    def __init__(self, name="Classifier"):
        super(Classifier, self).__init__(name=name)

    def compile(
        self, 
        losses=SoftmaxCrossEntropy(),
        loss_weights=None,
        optimizer="Adam",
        metrics=CategoricalAccuracy(),
        decoders=ArgMaxDecoder(),
        collate_fn=CollateList(inputs_axis=[0], targets_axis=[1])
    ):

        super(Classifier, self).compile(
            losses=losses,
            loss_weights=loss_weights,
            optimizer=optimizer,
            metrics=metrics,
            decoders=decoders,
            collate_fn=collate_fn
        )

class Regressor(Model):

    def __init__(self, name="Regressor"):
        super(Regressor, self).__init__(name=name)

    def compile(
        self, 
        losses=MeanSquaredError(),
        loss_weights=None,
        optimizer="Adam",
        metrics="MeanAbsoluteError",
        decoders=None,
        collate_fn=CollateList(inputs_axis=[0], targets_axis=[1])
    ):

        super(Regressor, self).compile(
            losses=losses,
            loss_weights=loss_weights,
            optimizer=optimizer,
            metrics=metrics,
            decoders=decoders,
            collate_fn=collate_fn
        )