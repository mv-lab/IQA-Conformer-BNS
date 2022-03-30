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

# Other
import sentencepiece as spm

# CTC Decode
try:
    from ctcdecode import CTCBeamDecoder
except Exception as e:
    pass
    #print(e)

###############################################################################
# Decoders
###############################################################################

class ArgMaxDecoder(nn.Module):

    def __init__(self, axis=-1):
        super(ArgMaxDecoder, self).__init__()
        self.axis = axis

    def forward(self, outputs, from_logits=True):

        if from_logits:
            # Softmax -> Log -> argmax
            tokens = outputs.softmax(dim=self.axis).argmax(axis=self.axis)#.tolist()
        else:
            tokens = outputs#.tolist()

        return tokens


###############################################################################
# Decoder Dictionary
###############################################################################

decoder_dict = {
    "ArgMax": ArgMaxDecoder,
}
