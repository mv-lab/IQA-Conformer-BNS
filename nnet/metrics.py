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
import jiwer
from scipy.stats import spearmanr

###############################################################################
# Metrics
###############################################################################

class Mean(nn.Module):

    def __init__(self, name="mean"):
        super(Mean, self).__init__()
        self.name = name

    def forward(self, y_true, y_pred):

        # To Tensor
        y_pred = torch.tensor(y_pred)

        # Compute mean
        mean = y_pred.mean()

        return mean

class MeanAbsoluteDist(nn.Module):

    def __init__(self, name="mad"):
        super(MeanAbsoluteDist, self).__init__()
        self.name = name

    def forward(self, y_true, y_pred):

        # Compute mean
        mean = (y_true - y_pred).abs().mean()

        return mean

class MeanSquaredDist(nn.Module):

    def __init__(self, name="msd"):
        super(MeanSquaredDist, self).__init__()
        self.name = name

    def forward(self, y_true, y_pred):

        # Compute mean
        mean = (y_true - y_pred).square().mean()

        return mean

class BinaryAccuracy(nn.Module):

    def __init__(self, ignore_index=-1, targets=None, name="acc"):
        super(BinaryAccuracy, self).__init__()
        self.ignore_index = ignore_index
        self.name = name
        if targets != None:
            self.register_buffer("targets", torch.tensor(targets))
        else:
            self.targets = None

    def forward(self, y_true, y_pred):

        # To Tensor
        y_pred = torch.tensor(y_pred)
        if self.targets != None:
            y_true = self.targets.expand_as(y_pred).to(y_pred.device)
        else:
            y_true = torch.tensor(y_true)
        
        # Compute Mask
        mask = torch.where(y_true==self.ignore_index, 0.0, 1.0)

        # Reduction
        n = torch.count_nonzero(mask)

        # Element Wise Accuracy
        acc = torch.where(y_true==y_pred, 1.0, 0.0)

        # Mask Accuracy
        acc = acc * mask

        # Binary Accuracy
        acc = 100 * acc.sum() / n

        return acc

class CategoricalAccuracy(nn.Module):

    def __init__(self, ignore_index=-1, name="acc"):
        super(CategoricalAccuracy, self).__init__()
        self.name = name
        self.ignore_index = ignore_index

    def forward(self, y_true, y_pred):

        # Compute Mask
        mask = torch.where(y_true==self.ignore_index, 0.0, 1.0)

        # Reduction
        n = torch.count_nonzero(mask)

        # Element Wise Accuracy
        acc = torch.where(y_true==y_pred, 1.0, 0.0)

        # Mask Accuracy
        acc = acc * mask

        # Categorical Accuracy
        acc = 100 * acc.sum() / n

        return acc

class WordErrorRate(nn.Module):

    def __init__(self, name="wer"):
        super(WordErrorRate, self).__init__()
        self.name = name

    def forward(self, targets, outputs):

        # Word Error Rate
        return 100 * jiwer.wer(targets, outputs, standardize=True)

class PLCC(nn.Module):

    """Pearson Linear Correlation Coefficient"""

    def __init__(self, name="plcc"):
        super(PLCC, self).__init__()
        self.name = name

    def forward(self, y_true, y_pred):

        # To Tensor
        #y_true = torch.tensor(y_true)
        #y_pred = torch.tensor(y_pred)

        y_true_center = y_true - y_true.mean()
        y_pred_denter = y_pred - y_pred.mean()

        num = (y_true_center * y_pred_denter).sum()
        den = (y_true_center.square().sum() * y_pred_denter.square().sum()).sqrt()
        
        plcc = num / den

        #plcc = max(min(plcc, 1), -1).abs()

        return plcc

class SROCC(nn.Module):

    """Spearman Rank-Order Correlation Coefficient"""

    def __init__(self, name="srocc"):
        super(SROCC, self).__init__()
        self.name = name

    def forward(self, y_true, y_pred):

        srocc = abs(spearmanr(y_true.cpu().type(torch.float32), y_pred.cpu().type(torch.float32))[0])

        #srocc = 1 - 6 * (torch.linalg.matrix_rank(y_true) - torch.linalg.matrix_rank(y_pred)).square() / (y_true.size(0) * (y_true.size(0) ** 2 - 1))

        return srocc

class PearsonSpearman(nn.Module):

    def __init__(self, name="ps"):
        super(PearsonSpearman, self).__init__()
        self.name = name
        self.plcc = PLCC()
        self.srocc = SROCC()

    def forward(self, y_true, y_pred):

        return self.plcc(y_true, y_pred) + self.srocc(y_true, y_pred)

class FrechetVideoDistance(nn.Module):

    """ Frechet Video Distance, Video FID

        Infos: The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1) and X_2 ~ N(mu_2, C_2) is
        d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)). 
    
        Reference: "Towards Accurate Generative Models of Video: A New Metric & Challenges", Unterthiner et al.
        https://arxiv.org/abs/1812.01717

    """

    def __init__(self, name="FVD", net="I3D"):
        super(FrechetVideoDistance, self).__init__()
        self.name = name
        self.net = None

    def forward(self, y_true, y_pred):

        # Compute logits
        logits_true = self.net(y_true)
        logits_pred = self.net(y_pred)

        # Flatten (B, ...) -> (B, N)
        logits_true = logits_true.flatten(start_dim=1, end_dim=-1)
        logits_pred = logits_pred.flatten(start_dim=1, end_dim=-1)

        # Mean (B, N) -> (B)
        mean_true = logits_true.mean(dim=1)
        mean_pred = logits_pred.mean(dim=1)

        # Covariance (B, N, N)
        cov_true = (logits_true - mean_true).unsqueeze(dim=-1).matmul((logits_true - mean_true).unsqueeze(dim=1))
        cov_pred = (logits_pred - mean_pred).unsqueeze(dim=-1).matmul((logits_pred - mean_pred).unsqueeze(dim=1))

        fvd_cov = (cov_true + cov_pred + 2 * (cov_true * cov_pred).sqrt()).trace()
        fvd_mean = (mean_true - mean_pred).square().sum()

        fvd = fvd_mean + fvd_cov

        return fvd



###############################################################################
# Metric Dictionary
###############################################################################

metric_dict = {
    "MeanAbsoluteDist": MeanAbsoluteDist,
    "MeanSquaredDist": MeanSquaredDist,
    "BinaryAccuracy": BinaryAccuracy,
    "CategoricalAccuracy": CategoricalAccuracy,
    "WordErrorRate": WordErrorRate
}