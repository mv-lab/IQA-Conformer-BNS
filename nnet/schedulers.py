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

import math
import torch

###############################################################################
# Schedulers
###############################################################################

class Scheduler(nn.Module):

    def __init__(self):
        super(Scheduler, self).__init__()

        # Model Step
        self.model_step = torch.tensor(0)

    def step(self):
        self.model_step += 1

class ConstantScheduler(Scheduler):

    def __init__(self, lr_value):
        super(ConstantScheduler, self).__init__()

        # Scheduler Params
        self.lr_value = lr_value

    def step(self):
        super(ConstantScheduler, self).step()

        # Return LR
        return self.lr_value

class ConstantDecayScheduler(Scheduler):

    def __init__(self, lr_values, decay_steps):
        super(ConstantDecayScheduler, self).__init__()

        # Scheduler Params
        self.lr_values = lr_values
        self.decay_steps = decay_steps

    def step(self):
        super(ConstantDecayScheduler, self).step()

        # Compute LR
        lr_value = self.lr_values[0]
        for i, step in enumerate(self.decay_steps):
            if self.model_step > step:
                lr_value = self.lr_values[i + 1]
            else:
                break
        return lr_value

class LinearDecayScheduler(Scheduler):

    def __init__(self, value_start, value_end, decay_steps):
        super(LinearDecayScheduler, self).__init__()

        # Scheduler Params
        self.value_start = value_start
        self.value_end = value_end
        self.decay_steps = decay_steps

    def step(self):
        super(LinearDecayScheduler, self).step()

        # Compute LR
        if self.model_step >= self.decay_steps:
            value = self.value_end
        else:
            value = self.value_start - self.model_step * (self.value_start - self.value_end) / self.decay_steps

        return value

class NoamDecayScheduler(Scheduler):

    def __init__(self, warmup_steps, dim_decay, lr_factor):
        super(NoamDecayScheduler, self).__init__()

        # Scheduler Params
        self.warmup_steps = warmup_steps
        self.dim_decay = dim_decay
        self.lr_factor = lr_factor

    def step(self):
        super(NoamDecayScheduler, self).step()

        # Compute LR
        arg1 = self.model_step * (self.warmup_steps**-1.5) # Warmup phase
        arg2 = self.model_step**-0.5 # Decay phase
        return self.lr_factor * self.dim_decay**-0.5 * min(arg1, arg2)

class ExpDecayScheduler(Scheduler):

    def __init__(self, warmup_steps, lr_max, alpha, end_step):
        super(ExpDecayScheduler, self).__init__()

        # Scheduler Params
        self.warmup_steps = warmup_steps
        self.lr_max = lr_max
        self.alpha = alpha
        self.end_step = end_step

    def step(self):
        super(ExpDecayScheduler, self).step()

        # Compute LR
        arg1 = self.model_step / self.warmup_steps * self.lr_max # Warmup phase
        arg2 = self.lr_max * self.alpha**((self.model_step - self.warmup_steps) / (self.end_step - self.warmup_steps)) # Decay phase
        return min(arg1, arg2)

class CosineAnnealingScheduler(Scheduler):

    def __init__(self, warmup_steps, lr_max, lr_min, end_step):
        super(CosineAnnealingScheduler, self).__init__()

        # Scheduler Params
        self.warmup_steps = warmup_steps
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.end_step = end_step

    def step(self):
        super(CosineAnnealingScheduler, self).step()

        # Compute LR
        if self.model_step <= self.warmup_steps: # Warmup phase
            return self.model_step / self.warmup_steps * self.lr_max
        elif self.model_step <= self.end_step: # Annealing phase
            return (self.lr_max - self.lr_min) * 0.5 * (1 + math.cos(math.pi * (self.model_step - self.warmup_steps) / (self.end_step - self.warmup_steps))) + self.lr_min
        else: # End phase
            return self.lr_min

###############################################################################
# Scheduler Dictionary
###############################################################################

scheduler_dict = {
    "Constant": ConstantScheduler,
    "ConstantDecay": ConstantDecayScheduler,
    "NoamDecay": NoamDecayScheduler,
    "ExpDecay": ExpDecayScheduler,
    "CosineAnnealing": CosineAnnealingScheduler
}