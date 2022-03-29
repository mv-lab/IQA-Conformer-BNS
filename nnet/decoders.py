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

class IdentityDecoder(nn.Module):

    def __init__(self):
        super(IdentityDecoder, self).__init__()

    def forward(self, outputs, from_logits=True):

        return outputs.tolist()

class ThresholdDecoder(nn.Module):

    def __init__(self, threshold=0.5):
        super(ThresholdDecoder, self).__init__()
        self.threshold = threshold

    def forward(self, outputs, from_logits=True):

        if from_logits:
            tokens = torch.where(outputs >= self.threshold, 1, 0).squeeze(dim=-1).tolist()
        else:
            tokens = outputs.tolist()

        return tokens

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

class CTCGreadySearchDecoder(nn.Module):

    def __init__(self, tokenizer_path):
        super(CTCGreadySearchDecoder, self).__init__()

        # Load Tokenizer
        self.tokenizer = spm.SentencePieceProcessor(tokenizer_path)

    def forward(self, outputs, from_logits=True):

        if from_logits:
            tokens = self.gready_search(outputs["samples"], outputs["lengths"])
        else:
            tokens = outputs["samples"].tolist()

        return self.tokenizer.decode(tokens)

    def gready_search(self, logits, logits_len):

        # Softmax -> Log -> Argmax -> (B, T)
        preds = logits.log_softmax(dim=-1).argmax(dim=-1)

        # Batch Pred List
        batch_pred_list = []

        # Batch loop
        for b in range(logits.size(0)):

            # Blank
            blank = False

            # Pred List
            pred_list = []

            # Decoding Loop
            for t in range(logits_len[b]):

                # Blank Prediction
                if preds[b, t] == 0:
                    blank = True
                    continue

                # First Prediction
                if len(pred_list) == 0:
                    pred_list.append(preds[b, t].item())

                # New Prediction
                elif pred_list[-1] != preds[b, t] or blank:
                    pred_list.append(preds[b, t].item())
                
                # Update Blank
                blank = False

            # Append Sequence
            batch_pred_list.append(pred_list)

        return batch_pred_list

class CTCBeamSearchDecoder(nn.Module):

    def __init__(self, tokenizer_path, beam_size, tmp, ngram_path, ngram_alpha, ngram_beta, ngram_offset):
        super(CTCBeamSearchDecoder, self).__init__()

        # Load Tokenizer
        self.tokenizer = spm.SentencePieceProcessor(tokenizer_path)

        # Params
        self.beam_size = beam_size
        self.tmp = tmp
        self.ngram_path = ngram_path
        self.ngram_alpha = ngram_alpha
        self.ngram_beta = ngram_beta
        self.ngram_offset = ngram_offset

    def forward(self, outputs, from_logits=True):

        if from_logits:
            tokens = self.beam_search(outputs["samples"], outputs["lengths"])
        else:
            tokens = outputs["samples"].tolist()

        return self.tokenizer.decode(tokens)

    def beam_search(self, logits, logits_len):

        # Beam Search Decoder
        decoder = CTCBeamDecoder(
            [chr(idx + self.ngram_offset) for idx in range(self.tokenizer.vocab_size())],
            model_path=self.ngram_path,
            alpha=self.ngram_alpha,
            beta=self.ngram_beta,
            cutoff_top_n=self.tokenizer.vocab_size(),
            cutoff_prob=1.0,
            beam_width=self.beam_size,
            num_processes=8,
            blank_id=0,
            log_probs_input=True
        )

        # Apply Temperature
        logits = logits / self.tmp

        # Softmax -> Log
        logP = logits.log_softmax(dim=-1)

        # Beam Search Decoding
        beam_results, beam_scores, timesteps, out_lens = decoder.decode(logP, logits_len)

        # Batch Pred List
        batch_pred_list = []

        # Batch loop
        for b in range(logits.size(0)):
            batch_pred_list.append(beam_results[b][0][:out_lens[b][0]].tolist())

        return batch_pred_list

###############################################################################
# Decoder Dictionary
###############################################################################

decoder_dict = {
    "Threshold": ThresholdDecoder,
    "ArgMax": ArgMaxDecoder,
    "CTCGreadySearch": CTCGreadySearchDecoder,
    "CTCBeamSearch": CTCBeamSearchDecoder
}
