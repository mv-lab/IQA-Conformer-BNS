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
import torchaudio

###############################################################################
# Losses
###############################################################################

class Reduction(nn.Module):

    def __init__(self, reduction="mean"):
        super(Reduction, self).__init__()

        assert reduction in ["sum", "mean", "mean_batch"]
        self.reduction = reduction

    def forward(self, x, n_elt=None):

        # Reduction
        if self.reduction == "sum":
            x = x.sum()
        elif self.reduction == "mean" and n_elt == None:
            x = x.mean()
        elif self.reduction == "mean" and n_elt != None:
            x = x.sum() / n_elt
        elif self.reduction == "mean_batch":
            x = x.mean(dim=0).sum()

        return x

class MeanLoss(nn.Module):

    def __init__(self, targets_as_sign=True, targets=None, reduction="mean"):
        super(MeanLoss, self).__init__()
        
        self.targets_as_sign = targets_as_sign
        if targets != None:
            self.register_buffer("targets", torch.tensor(targets))
        else:
            self.targets = None

        # Reduction
        self.reduction = Reduction(reduction)

    def forward(self, targets, outputs):

        # Unpack Outputs
        y_pred = outputs

        # Unpack Targets
        if self.targets != None:
            y = self.targets.expand_as(y_pred).to(y_pred.device)
        else:
            y = targets
        
        # Loss Sign
        if self.targets_as_sign:
            y_pred = torch.where(y == 1, - y_pred, y_pred)

        # Reduction
        loss = self.reduction(y_pred)

        return loss

class MeanAbsoluteError(nn.L1Loss):

    def __init__(self, convert_one_hot=False, one_hot_axis=-1, masked=False, reduction='mean'):
        super(MeanAbsoluteError, self).__init__()

        # Params
        self.convert_one_hot = convert_one_hot
        self.one_hot_axis = one_hot_axis
        self.masked = masked

        # Loss
        self.loss = nn.L1Loss(reduction='none')

        # Reduction
        self.reduction = Reduction(reduction)

    def forward(self, targets, outputs):

        # Unpack Targets
        y = targets

        # Unpack Outputs
        if self.masked:
            y_pred, mask = outputs
        else:
            y_pred = outputs

        # Convert one hot
        if self.convert_one_hot:
            y = F.one_hot(y, num_classes=y_pred.size(self.one_hot_axis)).type(y_pred.dtype)

        # Compute Loss
        loss = self.loss(input=y_pred, target=y)

        # Mask Loss
        if self.masked:
            loss = loss * mask
            N = mask.count_nonzero()
        else:
            N = loss.numel()

        # Reduction
        loss = self.reduction(loss, n_elt=N)

        return loss

class MeanSquaredError(nn.MSELoss):

    def __init__(self, convert_one_hot=False, axis=-1, targets=None, reduction='mean'):
        super(MeanSquaredError, self).__init__()

        # Params
        self.convert_one_hot = convert_one_hot
        self.axis = axis

        # Targets
        if targets != None:
            self.register_buffer("targets", torch.tensor(targets))
        else:
            self.targets = None

        # Loss
        self.loss = nn.MSELoss(reduction='none')

        # Reduction
        self.reduction = Reduction(reduction)

    def forward(self, targets, outputs):

        # Unpack Outputs
        y_pred = outputs

        # Unpack Targets
        if self.targets != None:
            y = self.targets.expand_as(y_pred).to(y_pred.device)
        else:
            y = targets

        # Convert one hot
        if self.convert_one_hot:
            y = F.one_hot(y, num_classes=y_pred.size(self.axis)).type(y_pred.dtype)

        # Compute Loss
        loss = self.loss(input=y_pred, target=y)

        # Reduction
        loss = self.reduction(loss)

        return loss

class HuberLoss(nn.HuberLoss):

    def __init__(self, convert_one_hot=False, axis=-1, targets=None, reduction='mean'):
        super(HuberLoss, self).__init__(reduction='none', delta=1.0)
        self.convert_one_hot = convert_one_hot
        self.axis = axis

        if targets != None:
            self.register_buffer("targets", torch.tensor(targets))
        else:
            self.targets = None

        # Reduction
        assert reduction in ["sum", "mean", "mean_batch"]
        self.red = reduction

    def forward(self, targets, outputs):

        # Unpack Outputs
        y_pred = outputs

        # Unpack Targets
        if self.targets != None:
            y = self.targets.expand_as(y_pred).to(y_pred.device)
        else:
            y = targets

        # Convert one hot
        if self.convert_one_hot:
            y = F.one_hot(y, num_classes=y_pred.size(self.axis)).type(y_pred.dtype)

        # Compute Loss
        loss = super(HuberLoss, self).forward(
            input=y_pred,
            target=y
        )

        # Reduction
        if self.red == "sum":
            loss = loss.sum()
        elif self.red == "mean":
            loss = loss.mean()
        elif self.red == "mean_batch":
            loss = loss.mean(dim=0).sum()

        return loss

class BinaryCrossEntropy(nn.BCELoss):

    def __init__(self, squeeze_pred=False, label_smoothing=0.0, one_sided=False, targets=None, reduction='mean'):
        super(BinaryCrossEntropy, self).__init__(weight=None, size_average=None, reduce=None, reduction="none")
        self.squeeze_pred = squeeze_pred
        self.label_smoothing = label_smoothing
        self.one_sided = one_sided
        if targets != None:
            self.register_buffer("targets", torch.tensor(targets))
        else:
            self.targets = None

        # Reduction
        assert reduction in ["sum", "mean", "mean_batch"]
        self.red = reduction

    def forward(self, targets, outputs):

        # Unpack Outputs
        y_pred = outputs

        # Unpack Targets
        if self.targets != None:
            y = self.targets.expand_as(y_pred).to(y_pred.device)
        else:
            y = targets

        # Squeeze pred
        if self.squeeze_pred:
            y_pred = y_pred.squeeze(dim=-1)

        # Label Smoothing
        if self.one_sided:
            y = (1 - self.label_smoothing) * y
        else:
            y = (1 - self.label_smoothing) * y + self.label_smoothing / 2

        # Compute Loss
        loss = super(BinaryCrossEntropy, self).forward(
            input=y_pred,
            target=y
        )

        # Reduction
        if self.red == "sum":
            loss = loss.sum()
        elif self.red == "mean":
            loss = loss.mean()
        elif self.red == "mean_batch":
            loss = loss.mean(dim=0).sum()

        return loss

class SigmoidBinaryCrossEntropy(nn.BCEWithLogitsLoss):

    def __init__(self, squeeze_logits=True):
        super(SigmoidBinaryCrossEntropy, self).__init__(weight=None, size_average=None, reduce=None, reduction='mean', pos_weight=None)
        self.squeeze_logits = squeeze_logits

    def forward(self, targets, outputs):

        # Unpack Targets
        y = targets

        # Unpack Outputs
        logits = outputs

        # Squeeze logits
        if self.squeeze_logits:
            logits = logits.squeeze(dim=-1)

        # Compute Loss
        loss = super(SigmoidBinaryCrossEntropy, self).forward(
            input=logits,
            target=y
        )

        return loss

class SoftmaxCrossEntropy(nn.CrossEntropyLoss):

    def __init__(self, ignore_index=-1, transpose_logits=False, reduction='mean'):
        super(SoftmaxCrossEntropy, self).__init__(weight=None, size_average=None, ignore_index=ignore_index, reduce=None, reduction='none')
        self.transpose_logits = transpose_logits

        # Reduction
        assert reduction in ["sum", "mean", "mean_batch"]
        self.red = reduction

    def forward(self, targets, outputs):

        # Unpack Targets
        y = targets

        # Unpack Outputs
        logits = outputs

        # transpose Logits
        if self.transpose_logits:
            logits = logits.transpose(1, 2)

        # Compute Loss
        loss = super(SoftmaxCrossEntropy, self).forward(
            input=logits,
            target=y
        )

        # Reduction
        if self.red == "sum":
            loss = loss.sum()
        elif self.red == "mean":
            loss = loss.mean()
        elif self.red == "mean_batch":
            loss = loss.mean(dim=0).sum()

        return loss

class NegativeLogLikelihood(nn.Module):

    """Elementwise Negative Log Likelihood"""

    def __init__(self, reduction='mean'):
        super(NegativeLogLikelihood, self).__init__()

        # Reduction
        assert reduction in ["sum", "mean", "mean_batch"]
        self.red = reduction

    def forward(self, targets, outputs):

        # Unpack Targets
        y = targets

        # Unpack Outputs
        y_pred = outputs

        # Compute Loss
        loss = - y * y_pred.log()

        # Reduction
        if self.red == "sum":
            loss = loss.sum()
        elif self.red == "mean":
            loss = loss.mean()
        elif self.red == "mean_batch":
            loss = loss.mean(dim=0).sum()

        return loss

class CTCLoss(nn.CTCLoss):

    def __init__(self):
        super(CTCLoss, self).__init__(blank=0, reduction="none", zero_infinity=False)

    def forward(self, targets, outputs):

        # Unpack Targets
        y, y_len = targets

        # Unpack Outputs
        logits, logits_len = outputs

        # Compute Loss
        loss = super(CTCLoss, self).forward(
             log_probs=torch.nn.functional.log_softmax(logits, dim=-1).transpose(0, 1),
             targets=y,
             input_lengths=logits_len,
             target_lengths=y_len
        ).mean()

        return loss

class RNNTLoss(torchaudio.transforms.RNNTLoss):

    def __init__(self, blank=0, clamp=-1, reduction="mean"):
        super(RNNTLoss, self).__init__(blank=blank, clamp=clamp, reduction=reduction)

    def forward(self, targets, outputs):

        # Unpack Targets (B, U) and (B,)
        y, y_len = targets

        # Unpack Outputs (B, T, U + 1, V) and (B,)
        logits, logits_len = outputs

        #print(logits.size(), logits_len)
        #print(y.size(), y_len)

        # Compute Loss
        loss = super(RNNTLoss, self).forward(
            logits=logits,
            targets=y.int(),
            logit_lengths=logits_len.int(),
            target_lengths=y_len.int()
        )

        return loss

class KullbackLeiblerDivergence(nn.Module):

    """Kullback Leibler Divergence
    
    Relative entropy between a diagonal multivariate normal, 
    and a standard normal distribution (with zero mean and unit variance)
    
    """

    def __init__(self, axis=-1):
        super(KullbackLeiblerDivergence, self).__init__()
        self.axis = axis

    def forward(self, targets, outputs):

        # Unpack Outputs
        mean, log_var = outputs

        loss = 0.5 * (torch.square(mean) + torch.exp(log_var) - 1 - log_var).sum(axis=self.axis)

        # Reduction
        loss = loss.mean()

        return loss

class NegCosSim(nn.Module):

    """ Negative Cosine Similarity
    
    Info:
        Negative Cosine of the angle between two vectors, that is, the dot product of the vectors divided by the product of their lengths.
        Similarity belongs to the interval [-1,1]

    """

    def __init__(self, dim: int = 1, eps: float = 1e-8):
        super(NegCosSim, self).__init__()

        # Loss
        self.sim = nn.CosineSimilarity(dim=dim, eps=eps)

    def forward(self, targets, outputs):

        # Compute loss
        loss = 1 - self.sim(x1=targets, x2=outputs)

        # Reduction
        loss = loss.mean()

        return loss

class GradientPenalty(nn.Module):

    """Compute the Gradient Penalty Loss

    Info:
        Enforce differentiable function to have gradients norm close to 1

    References:
        Improved Training of Wasserstein GANs, Gulrajani et al.
        https://arxiv.org/abs/1704.00028
    
    """

    def __init__(self, net):
        super(GradientPenalty, self).__init__()
        self.net = net

    def forward(self, targets, outputs):

        # Unpack Targets
        real = targets.data

        # Unpack Outputs
        fake = outputs.data

        # Sample alpha 
        alpha = torch.rand(real.size(0), 1, device=real.device).expand(real.size(0), real.nelement() // real.size(0)).reshape(real.size())

        # Interpolates
        interpolates = alpha * real + (1 - alpha) * fake

        # Requires Grad
        interpolates.requires_grad_(True)

        # Forward Network
        preds = self.net(interpolates)

        # Compute Gradients
        gradients = torch.autograd.grad(
            outputs=preds, 
            inputs=interpolates,
            grad_outputs=preds.new_ones(preds.size()),
            create_graph=True, 
            retain_graph=True, 
            only_inputs=True
        )[0]

        # Flatten Gradients
        gradients = gradients.flatten(start_dim=1, end_dim=-1)

        # Compute Gradient Penalty
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

        return gradient_penalty

class PerceptualLoss(nn.Module):

    def __init__(self, net):
        super(PerceptualLoss, self).__init__()

        # Network
        self.net = net

    def forward(self, targets, outputs):

        pass

class BinaryCrossEntropyRCNN(nn.BCELoss):

    def __init__(self):
        super(BinaryCrossEntropyRCNN, self).__init__(weight=None, size_average=None, reduce=None, reduction="none")

    def forward(self, targets, outputs):

        # Unpack Outputs
        y_pred = outputs

        # Unpack Targets
        y, mask = targets

        # Compute Loss
        loss = super(BinaryCrossEntropyRCNN, self).forward(
            input=y_pred,
            target=y
        )

        # Apply Mask
        loss *= mask

        # Number Valid Boxes
        N = mask.count_nonzero()

        # Reduction
        loss = loss.sum() / N

        return loss

class SmoothL1RCNN(nn.SmoothL1Loss):

    def __init__(self):
        super(SmoothL1RCNN, self).__init__(reduction='none', beta=1.0)

    def forward(self, targets, outputs):

        # Unpack Outputs
        y_pred = outputs

        # Unpack Targets
        y, mask = targets

        # Compute Loss
        loss = super(SmoothL1RCNN, self).forward(
            input=y_pred,
            target=y
        )

        # Apply Mask
        loss *= mask

        # Number Valid Boxes
        N = mask.count_nonzero()

        # Reduction
        loss = loss.sum() / N

        return loss

class PearsonLoss(nn.Module):

    """Pearson Linear Correlation Coefficient Loss"""

    def __init__(self):
        super(PearsonLoss, self).__init__()

    def forward(self, targets, outputs):

        # Unpack Outputs
        y_pred = outputs

        # Unpack Targets
        y_true = targets

        y_true_center = y_true - y_true.mean()
        y_pred_denter = y_pred - y_pred.mean()

        num = (y_true_center * y_pred_denter).sum()
        den = (y_true_center.square().sum() * y_pred_denter.square().sum()).sqrt()
        
        loss = num / den

        # loss ∈ [0:1]
        loss = 1 - loss.square()

        return loss


class LossInterCTC(nn.Module):

    def __init__(self, interctc_lambda):
        super(LossInterCTC, self).__init__()

        # CTC Loss
        self.loss = nn.CTCLoss(blank=0, reduction="none", zero_infinity=False)

        # InterCTC Lambda
        self.interctc_lambda = interctc_lambda

    def forward(self, batch, pred):

        # Unpack Batch
        x, y, x_len, y_len = batch

        # Unpack Predictions
        outputs_pred, f_len, _, interctc_probs = pred

        # Compute CTC Loss
        loss_ctc = self.loss(
             log_probs=torch.nn.functional.log_softmax(outputs_pred, dim=-1).transpose(0, 1),
             targets=y,
             input_lengths=f_len,
             target_lengths=y_len)

        # Compute Inter Loss
        loss_inter = sum(self.loss(
             log_probs=interctc_prob.log().transpose(0, 1),
             targets=y,
             input_lengths=f_len,
             target_lengths=y_len) for interctc_prob in interctc_probs) / len(interctc_probs)

        # Compute total Loss
        loss = (1 - self.interctc_lambda) * loss_ctc + self.interctc_lambda * loss_inter
        loss = loss.mean()

        return loss

###############################################################################
# Loss Dictionary
###############################################################################

loss_dict = {
    "Mean": MeanLoss,
    "MeanAbsoluteError": MeanAbsoluteError,
    "MeanSquaredError": MeanSquaredError,
    "HuberLoss": HuberLoss,
    "BinaryCrossEntropy": BinaryCrossEntropy,
    "SigmoidBinaryCrossEntropy": SigmoidBinaryCrossEntropy,
    "SoftmaxCrossEntropy": SoftmaxCrossEntropy,
    "CTC": CTCLoss,
    "RNNTLoss": RNNTLoss
}