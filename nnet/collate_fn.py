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

class MultiCollate(nn.Module):

    def __init__(self, dict_collate_fn):
        super(MultiCollate, self).__init__()

        self.dict_collate_fn = dict_collate_fn

    def forward(self, samples):

        # Init Dict
        collate_dict = {"inputs": {}, "targets":{}}

        # Collates Functions
        for key, value in self.dict_collate_fn.items():
            inputs, targets = value(samples).values()
            collate_dict["inputs"][key] = inputs
            collate_dict["targets"][key] = targets

        return collate_dict

class CollateGAN(nn.Module):

    def __init__(self, dis_collate, gen_collate):
        super(CollateGAN, self).__init__()

        self.dis_collate = dis_collate
        self.gen_collate = gen_collate

    def forward(self, samples):

        # Discriminator collate
        dis_inputs, dis_targets = self.dis_collate(samples).values()

        # Generator Collate
        gen_inputs, gen_targets = self.gen_collate(samples).values()

        return {"inputs": {"discriminator": dis_inputs, "generator": gen_inputs}, "targets": {"discriminator": dis_targets, "generator": gen_targets}}

class CollateDict(nn.Module):

    def __init__(self, inputs, targets):
        super(CollateDict, self).__init__()

        axis = 0

        # Inputs Params
        self.inputs = inputs
        for params in self.inputs.values():

            if params.get("noise", False):
                continue
            else:
                params["noise"] = False

            if not "start_token" in params:
                params["start_token"] = None
            if not "axis" in params:
                params["axis"] = axis
            if not "lengths" in params:
                params["lengths"] = False
            if not "padding" in params:
                params["padding"] = False
            if not "padding_token" in params:
                params["padding_token"] = 0

            axis += 1

        # Targets
        self.targets = targets
        for params in self.targets.values():
            if not "stop_token" in params:
                params["stop_token"] = None
            if not "axis" in params:
                params["axis"] = axis
            if not "lengths" in params:
                params["lengths"] = False
            if not "padding" in params:
                params["padding"] = False
            if not "padding_token" in params:
                params["padding_token"] = 0

            axis += 1

    def forward(self, samples):

        # Init
        inputs = {}
        targets = {}

        # Inputs
        for name, params in self.inputs.items():

            # Noise
            if params["noise"]:
                inputs[name] = torch.randn(params["batch_size"], params["dim"])
                continue

            # Start Token 
            if params["start_token"] != None:
                inputs_samples = [nn.functional.pad(sample[params["axis"]], pad=(1, 0), value=params["start_token"]) for sample in samples]
            else:
                inputs_samples = [sample[params["axis"]] for sample in samples]

            # Lengths
            if params["lengths"]:
                inputs[name + "_lengths"] = torch.tensor([len(sample) for sample in inputs_samples], dtype=torch.long)

            # Padding
            if params["padding"]:
                inputs[name] = torch.nn.utils.rnn.pad_sequence(inputs_samples, batch_first=True, padding_value=params["padding_token"])
            else:
                inputs[name] = torch.stack(inputs_samples, axis=0)

        # Targets
        for name, params in self.targets.items():

            # Stop Token
            if params["stop_token"] != None:
                targets_samples = [nn.functional.pad(sample[params["axis"]], pad=(0, 1), value=params["stop_token"]) for sample in samples]
            else:
                targets_samples = [sample[params["axis"]] for sample in samples]

            # Lengths
            if params["lengths"]:
                targets[name + "_lengths"] = torch.tensor([len(sample) for sample in targets_samples], dtype=torch.long)

            # Padding
            if params["padding"]:
                targets[name] = torch.nn.utils.rnn.pad_sequence(targets_samples, batch_first=True, padding_value=params["padding_token"])
            else:
                targets[name] = torch.stack(targets_samples, axis=0)

        #print({"inputs": inputs, "targets": targets})
        #exit()

        return {"inputs": inputs, "targets": targets}

class collate_fn0(nn.Module):

    def __init__(self, inputs_axis=0, targets_axis=1, inputs_padding=False, targets_padding=False, inputs_lengths=False, targets_lengths=False, start_token=None, stop_token=None, padding_token=0, sort_by_length=False):
        super(collate_fn0, self).__init__()

        self.inputs_padding = inputs_padding
        self.inputs_lengths = inputs_lengths
        self.inputs_axis = inputs_axis
        self.start_token = start_token

        self.targets_padding = targets_padding
        self.targets_lengths = targets_lengths
        self.targets_axis = targets_axis
        self.stop_token = stop_token

        self.padding_token = padding_token
        self.sort_by_length = sort_by_length

    def forward(self, samples):

        # Init
        inputs = {}
        targets = {}

        # Inputs
        # Start Token 
        if self.start_token != None:
            inputs_samples = [nn.functional.pad(sample[self.inputs_axis], pad=(1, 0), value=self.start_token) for sample in samples]
        else:
            inputs_samples = [sample[self.inputs_axis] for sample in samples]
        # Lengths
        if self.inputs_lengths:
            inputs["lengths"] = torch.tensor([len(sample) for sample in inputs_samples], dtype=torch.long)
        # Padding
        if self.inputs_padding:
            inputs["samples"] = torch.nn.utils.rnn.pad_sequence(inputs_samples, batch_first=True, padding_value=self.padding_token)
        else:
            inputs["samples"] = torch.stack([sample[self.inputs_axis] for sample in samples], axis=0)

        # Targets
        # Stop Token
        if self.stop_token != None:
            targets_samples = [nn.functional.pad(sample[self.targets_axis], pad=(0, 1), value=self.stop_token) for sample in samples]
        else:
            targets_samples = [sample[self.targets_axis] for sample in samples]
        # Lengths
        if self.targets_lengths:
            targets["lengths"] = torch.tensor([len(sample) for sample in targets_samples], dtype=torch.long)
        # Padding
        if self.targets_padding:
            targets["samples"] = torch.nn.utils.rnn.pad_sequence(targets_samples, batch_first=True, padding_value=self.padding_token)
        else:
            targets["samples"] = torch.stack([sample[self.targets_axis] for sample in samples], axis=0)

        return {"inputs": inputs, "targets": targets}