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
import nnet

# Other
import matplotlib.pyplot as plt

# Base Model
from nnet.model import Model

from nnet.quantizers import (
    VectorQuantizer,
    VectorQuantizerEMA,
    GumbelSoftmaxQuantizer
)

from nnet.buffers import (
    ReplayBuffer
)

# Layers
from nnet.layers import (
    Linear,
    Conv2d
)

# Losses
from nnet.losses import (
    MeanLoss,
    MeanAbsoluteError,
    MeanSquaredError,
    HuberLoss,
    BinaryCrossEntropy,
    SigmoidBinaryCrossEntropy,
    SoftmaxCrossEntropy,
    NegativeLogLikelihood,
    KullbackLeiblerDivergence,
    CTCLoss,
    RNNTLoss,
    BinaryCrossEntropyRCNN,
    SmoothL1RCNN
)

# Decoders
from nnet.decoders import (
    ThresholdDecoder,
    ArgMaxDecoder
)

# Metrics
from nnet.metrics import (
    Mean,
    BinaryAccuracy,
    CategoricalAccuracy,
    WordErrorRate
)

# Collate Functions
from nnet.collate_fn import (
    CollateList,
    MultiCollate
)

# Optimizers
from nnet.optimizers import (
    SGD,
    Adam
)

# Shedulers
from nnet.schedulers import (
    ConstantScheduler,
    LinearDecayScheduler,
    ConstantDecayScheduler
)

# Noises
from nnet.noises import (
    OrnsteinUhlenbeckProcess
)

# Networks Dict
from nnet.networks import networks_dict
networks_dict.update(nnet.blocks.block_dict)
networks_dict.update(nnet.modules.module_dict)
networks_dict.update(nnet.layers.layer_dict)
networks_dict.update(nnet.activations.act_dict)
networks_dict.update(nnet.normalizations.norm_dict)



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


class SequenceToSequenceModel(Model):

    def __init__(self, encoder, decoder, name="Sequence To Sequence Model"):
        super(Classifier, self).__init__(name=name)

        # Encoder
        self.encoder = encoder

        # Decoder
        self.decoder = decoder

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

    def forward(self, inputs):

        # Unpack Inputs
        x_enc, x_dec = inputs

        # Forward Encoder
        x_enc = self.encoder(x_enc)

        # Forward decoder
        outputs = self.decoder(x_enc, x_dec)

        return outputs

class FasterRCNN(Model):

    def __init__(self, 
        net, 

        anchor_scales=[64, 128, 256, 512], 
        anchor_ratios=[[1, 1], [1, 2], [2, 1]], 
        rpn_min_overlap=0.3, 
        rpn_max_overlap=0.7, 

        num_anchors=12, 
        pre_nms_limit=6000,
        nms_max_overlap=0.7, 
        post_nms_limit=2000,
        test_post_nms_limit=300,
        rois_per_image=300, 
        roi_pos_ratio=0.33,


        head_min_overlap=0.1,
        head_max_overlap=0.5,
        output_size=(7, 7), 
        dim_hidden=512, 
        dim_fc=4096, 
        dim_out=81, 

        rpn_batch_size=256,
        name="Faster Region CNN"
    ):
        super(FasterRCNN, self).__init__(name=name)

        # Base + RPN Network
        self.model_rpn = self.RPN(net, dim_hidden, num_anchors, pre_nms_limit, nms_max_overlap, post_nms_limit, roi_pos_ratio)

        # Head
        self.model_head = self.Head(output_size, dim_hidden, dim_fc, dim_out)

    def compile(
        self, 

        rpn_optimizer="SGD", 
        rpn_losses=[BinaryCrossEntropyRCNN(), SmoothL1RCNN()], 
        rpn_loss_weights=[1, 1], 
        rpn_metrics=[BinaryAccuracy(), None],
        rpn_decoders=None,
        rpn_collate_fn=CollateList(inputs_axis=[0], targets_axis=[1, 2]),

        head_optimizer="SGD",
        head_losses=[SoftmaxCrossEntropy(), SmoothL1RCNN()], 
        head_loss_weights=[1, 1],
        head_metrics=[CategoricalAccuracy(), None],
        head_decoders=None,
        head_collate_fn=CollateList(inputs_axis=[], targets_axis=[3, 2])
    ):

        # Compile RPN Model
        self.model_rpn.compile(
            optimizer=SGD(params=self.model_rpn.parameters(), lr=ConstantDecayScheduler(lr_values=[0.003, 0.0003], decay_steps=[240000, 80000]), momentum=0.9, weight_decay=0.0005) if rpn_optimizer == "SGD" else rpn_optimizer,
            losses=rpn_losses,
            loss_weights=rpn_loss_weights,
            metrics=rpn_metrics,
            decoders=rpn_decoders,
            collate_fn=None
        )

        # Compile Head Model
        self.model_head.compile(
            optimizer=SGD(self.model_head.parameters(), lr=ConstantDecayScheduler(lr_values=[0.003, 0.0003], decay_steps=[240000, 80000]), momentum=0.9, weight_decay=0.0005) if head_optimizer == "SGD" else head_optimizer,
            losses=head_losses,
            loss_weights=head_loss_weights,
            metrics=head_metrics,
            decoders=head_decoders,
            collate_fn=None
        )

        # Model Step
        self.model_step = self.model_rpn.optimizer.scheduler.model_step

        # Collate Function
        self.collate_fn = MultiCollate({"rpn": rpn_collate_fn, "head": head_collate_fn})

        # Optimizer: To Review
        self.optimizer = {"rpn": self.model_rpn.optimizer, "head": self.model_head.optimizer}

        # Set Compiled to True
        self.compiled = True

    def train_step(self, inputs, targets, mixed_precision, grad_scaler, accumulated_steps, acc_step):

        # Init Dict
        batch_losses = {}
        batch_metrics = {}

        # RPN Step
        rpn_batch_losses, rpn_batch_metrics, _ = self.model_rpn.train_step(inputs["rpn"], targets["rpn"], mixed_precision, grad_scaler, accumulated_steps, acc_step)
        batch_losses.update({"rpn_" + key: value for key, value in rpn_batch_losses.items()})
        batch_metrics.update({"rpn_" + key: value for key, value in rpn_batch_metrics.items()})

        # Head Step
        head_batch_losses, head_batch_metrics, _ = self.model_head.train_step(self.model_rpn.outputs, targets["head"], mixed_precision, grad_scaler, accumulated_steps, acc_step)
        batch_losses.update({"head_" + key: value for key, value in head_batch_losses.items()})
        batch_metrics.update({"head_" + key: value for key, value in head_batch_metrics.items()})

        # Update Infos
        self.infos.update(self.model_rpn.infos)
        self.infos.update(self.model_head.infos)

        return batch_losses, batch_metrics, _

    class RPN(Model):

        def __init__(self, net, dim_hidden, num_anchors, pre_nms_limit, nms_max_overlap, post_nms_limit, roi_pos_ratio):
            super().__init__()

            # Base Model
            self.net = net
            
            # RPN
            self.rpn_conv = Conv2d(dim_hidden, dim_hidden, kernel_size=(3, 3))
            self.rpn_class = Conv2d(dim_hidden, num_anchors, kernel_size=(1, 1))
            self.rpn_regr = Conv2d(dim_hidden, 4 * num_anchors, kernel_size=(1, 1))

            # Params
            self.pre_nms_limit = pre_nms_limit
            self.nms_max_overlap = nms_max_overlap
            self.post_nms_limit = post_nms_limit
            self.roi_pos_ratio = roi_pos_ratio


        def filter(self, scores, boxes, deltas):

            # Flatten Scores (B, H/S, W/S, N) -> (B, H/W * W/S * N)
            scores = scores.flatten(start_dim=1, end_dim=-1)

            # Flatten Coord (B, H/S, W/S, 4 * N) -> (B, H/W * W/S * N, 4)
            boxes = boxes.reshape(scores.shape + (4,))
            deltas = deltas.reshape(boxes.shape)

            # Apply deltas to anchors to get refined anchors.
            boxes = self.apply_deltas(boxes, deltas)

            # Pre NMS Top k (B, H/W * W/S * N) -> (B, Kpre)
            scores, indices = x_class.topk(k=min(self.pre_nms_limit, scores.size(1)), dim=-1, largest=True, sorted=True)
            boxes = boxes.gather(dim=1, index=indices)

            # Clip to image boundaries. Since we're in normalized coordinates,
            # clip to 0..1 range. [batch, N, (y1, x1, y2, x2)]
            boxes = self.clip_boxes(boxes, boxes.new([0, 0, 1, 1]))

            # NMS (B, Kpre) -> (B, K)
            indices = torchvision.ops.batched_nms(boxes, x_class, idxs=None, iou_threshold=self.nms_max_overlap)
            scores = scores.gather(dim=1, index=indices)
            boxes = boxes.gather(dim=1, index=indices)

            # Padding
            padding = max(self.post_nms_limit - scores.size(1), 0)
            boxes = F.pad(boxes, ((0, padding), (0, 0)))

            # Post NMS Top k (B, K) -> (B, Kpost)
            x_class, indices = x_class.topk(k=self.post_nms_limit, dim=-1, largest=True, sorted=True)
            boxes = boxes.gather(dim=1, index=indices)

            return boxes

        def center_coord(self, boxes, stack=True):

            """
            Return center coord [N, (cx, cy, w, h)]
            given abs coord [N, (x1, y1, x2, y2)]
            """

            # Convert to cx, cy, w, h
            w = boxes[:, 2] - boxes[:, 0]
            h = boxes[:, 3] - boxes[:, 1]
            cx = boxes[:, 0] + 0.5 * w
            cy = boxes[:, 1] + 0.5 * h

            if stack:
                coords =  torch.stack([cx, cy, w, h], axis=1)
            else:
                coords = [cx, cy, w, h]

            return coords


        def apply_deltas(self, boxes, deltas):

            """
            Applies the given deltas to the given boxes.
            boxes: [N, (x1, y1, x2, y2)] boxes to update
            deltas: [N, (dx, dy, log(dw), log(dh))] refinements to apply
            """

            # Convert to cx, cy, w, h
            w = boxes[:, 2] - boxes[:, 0]
            h = boxes[:, 3] - boxes[:, 1]
            cx = boxes[:, 0] + 0.5 * w
            cy = boxes[:, 1] + 0.5 * h
            
            # Apply deltas
            cx += deltas[:, 0] * w
            cy += deltas[:, 1] * h
            w *= deltas[:, 2].exp()
            h *= deltas[:, 3].exp()
            
            # Convert back to x1, y1, x2, y2
            x1 = cx - 0.5 * w
            y1 = cy - 0.5 * h
            x2 = x1 + w
            y2 = y1 + h
            
            # Stack [N, (x1, y1, x2, y2)]
            return torch.stack([x1, y1, x2, y2], axis=1)

        def clip_boxes(boxes, window):

            """
            Clip to window boundaries.
            boxes: [N, (x1, y1, x2, y2)]
            window: [win_x1, win_y1, win_x2, win_y2]
            """

            # Split
            win_x1, win_y1, win_x2, win_y2 = window.tensor_split(indices_or_sections=4, dim=0)
            y1, x1, y2, x2 = boxes.tensor_split(indices_or_sections=4, dim=1)

            # Clip
            x1 = x1.min(win_x2).max(win_x1)
            y1 = y1.min(win_y2).max(win_y1)
            x2 = x2.min(win_x2).max(win_x1)
            y2 = y2.min(win_y2).max(win_y1)

            # Stack [N, (x1, y1, x2, y2)]
            clipped = torch.stack([x1, y1, x2, y2], axis=1)

            return clipped

        def forward(self, x):

            # Backbone
            x_h = self.net(x)

            # RPN Conv2d
            x_conv = self.rpn_conv(x_h).relu()

            # Scores (B, H/S, W/S, N)
            x_class = self.rpn_class(x_conv).sigmoid()

            # Bboxes (B, H/S, W/S, 4 * N)
            x_regr = self.rpn_regr(x_conv)

            # Apply ROI Filter
            self.outputs = [x_h, self.filter(x_class, x_regr)]

            return x_class, x_regr

    class Head(Model):

        def __init__(self, output_size, dim_hidden, dim_fc, dim_out):
            super().__init__()

            # Head FC
            self.head_fc = nn.Sequential(
                Linear(dim_hidden * torch.prod(torch.tensor(output_size)), dim_fc),
                nn.ReLU(),
                Linear(dim_fc, dim_fc),
                nn.ReLU()
            )

            # Head Class
            self.head_class = Linear(dim_fc, dim_out)

            # Head Regr
            self.head_regr = Linear(dim_fc, 4 * (dim_out-1))

            # Params
            self.output_size = output_size

        def forward(self, x, rois):

            # Pooled Region of Interets (K, C, P, P)
            x_rois = torchvision.ops.roi_pool(x, rois, output_size=self.output_size)

            # Flatten (K, C, P, P) -> (K, C * P * P)
            x_rois = x_rois.flatten(1, -1)

            # FC Layers (K, C * P * P) -> (K, D)
            x_rois = self.head_fc(x_rois)

            # Class Outputs (K, D) -> (K, Dout)
            x_class = self.head_class(x_rois)

            # Regr Outputs (K, D) -> (K, 4 * (Dout-1))
            x_regr = self.head_regr(x_rois)

            return x_class, x_regr

###############################################################################
# Generative Adversarial Network Models
###############################################################################

class GenerativeAdversarialNetwork(Model):

    def __init__(self, generator, discriminator, name="Generative Adversarial Network"):
        super(GenerativeAdversarialNetwork, self).__init__(name=name)

        # Discriminator
        self.discriminator = discriminator

        # Generator
        self.generator = generator

        # Discriminator GAN
        self.dis_gan = self.DiscriminatorGAN(generator=generator, discriminator=discriminator)

        # Generator GAN
        self.gen_gan = self.GeneratorGAN(generator=generator, discriminator=discriminator)

    def compile(
        self, 

        dis_optimizer="Adam", 
        dis_losses=[BinaryCrossEntropy(targets=1.0), BinaryCrossEntropy(targets=0.0)], 
        dis_loss_weights=[0.5, 0.5], 
        dis_metrics=[Mean(name="mean_real"), Mean(name="mean_fake")],#[[Mean(name="mean_real"), BinaryAccuracy(targets=1.0, name="acc_real")], [Mean(name="mean_fake"), BinaryAccuracy(targets=0.0, name="acc_fake")]],
        dis_decoders=None,#[[None, ThresholdDecoder()], [None, ThresholdDecoder()]],
        dis_collate_fn=CollateList(inputs_axis=[0], targets_axis=[]),

        gen_optimizer="Adam",
        gen_losses=BinaryCrossEntropy(targets=1.0),
        gen_loss_weights=None,
        gen_metrics=Mean(name="mean_fake"),#[[Mean(name="mean_fake"), BinaryAccuracy(targets=1.0, name="acc_fake")]],
        gen_decoders=None,#[[None, ThresholdDecoder()]],
        gen_collate_fn=CollateList(inputs_axis=[], targets_axis=[])
    ):

        # Compile Discriminator GAN
        self.dis_gan.compile(
            optimizer=Adam(params=[{"params": net.parameters()} for net in self.discriminator] if isinstance(self.discriminator, list) else self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999), eps=1e-8) if dis_optimizer == "Adam" else dis_optimizer,
            losses=dis_losses,
            loss_weights=dis_loss_weights,
            metrics=dis_metrics,
            decoders=dis_decoders,
            collate_fn=None
        )

        # Compile Generator GAN
        self.gen_gan.compile(
            optimizer=Adam(params=[{"params": net.parameters()} for net in self.generator] if isinstance(self.generator, list) else self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999), eps=1e-8) if gen_optimizer == "Adam" else gen_optimizer,
            losses=gen_losses,
            loss_weights=gen_loss_weights,
            metrics=gen_metrics,
            decoders=gen_decoders,
            collate_fn=None
        )

        # Model Step
        self.model_step = self.dis_gan.optimizer.scheduler.model_step

        # Collate Function
        self.collate_fn = CollateGAN(dis_collate=dis_collate_fn, gen_collate=gen_collate_fn)

        # Optimizer: To Review
        self.optimizer = {"dis": self.dis_gan.optimizer, "gen": self.gen_gan.optimizer}

        # Set Compiled to True
        self.compiled = True

    def summary(self, show_dict=False):

        print(self.name)
        print("Model Parameters:", self.num_params(self.generator) + self.num_params(self.discriminator))

        print("Generator Parameters:", self.num_params(self.generator))
        if show_dict:
            self.show_dict(self.generator)

        print("Discriminator Parameters:", self.num_params(self.discriminator))
        if show_dict:
            self.show_dict(self.discriminator)

    def train_step(self, inputs, targets, mixed_precision, grad_scaler, accumulated_steps, acc_step):

        # Init Dict
        batch_losses = {}
        batch_metrics = {}

        # Discriminator Step
        self.set_require_grad(self.discriminator, True)
        dis_batch_losses, dis_batch_metrics, _ = self.dis_gan.train_step(inputs["discriminator"], targets["discriminator"], mixed_precision, grad_scaler, accumulated_steps, acc_step)
        batch_losses.update({"dis_" + key: value for key, value in dis_batch_losses.items()})
        batch_metrics.update({"dis_" + key: value for key, value in dis_batch_metrics.items()})

        # Generator Step
        self.set_require_grad(self.discriminator, False)
        gen_batch_losses, gen_batch_metrics, _ = self.gen_gan.train_step(inputs["generator"], targets["generator"], mixed_precision, grad_scaler, accumulated_steps, acc_step)
        batch_losses.update({"gen_" + key: value for key, value in gen_batch_losses.items()})
        batch_metrics.update({"gen_" + key: value for key, value in gen_batch_metrics.items()})

        # Update Infos
        self.infos.update(self.dis_gan.infos)
        self.infos.update(self.gen_gan.infos)

        return batch_losses, batch_metrics, _

    class DiscriminatorGAN(Model):

        def __init__(self, generator, discriminator):
            super().__init__()
            self.generator = generator
            self.discriminator = discriminator

        def forward(self, inputs):

            # Unpack Inputs
            real = inputs

            # Forward Generator
            self.generator.fake = self.generator(real.size(0))

            # Forward Discriminator
            pred_real = self.discriminator(real)
            pred_fake = self.discriminator(self.generator.fake.detach())

            return {"real": pred_real, "fake": pred_fake, "GP": self.generator.fake}

    class GeneratorGAN(Model):

        def __init__(self, generator, discriminator):
            super().__init__()
            self.generator = generator
            self.discriminator = discriminator

        def forward(self, inputs=None):

            # Forward Discriminator
            preds = self.discriminator(self.generator.fake)

            return preds

    def on_epoch_end(self, saving_period, callback_path, epoch, inputs, targets):
        super(GenerativeAdversarialNetwork, self).on_epoch_end(saving_period, callback_path, epoch, inputs, targets)

        # Eval Mode
        self.eval()

        # Batch Size
        nrow = 10
        B = nrow**2

        # Forward Generator
        with torch.no_grad():
            gen_outputs = self.generator(B)

        # Save Image Grid
        torchvision.utils.save_image(gen_outputs, callback_path + "samples_" + str(epoch + 1) + ".png", nrow=nrow, normalize=True)

class ConditionalGAN(GenerativeAdversarialNetwork):

    def __init__(self, generator, discriminator, name="Conditional Generative Adversarial Network"):
        super(ConditionalGAN, self).__init__(generator=generator, discriminator=discriminator, name=name)

    def compile(
        self,

        dis_optimizer="Adam", 
        dis_losses=[BinaryCrossEntropy(targets=1.0), BinaryCrossEntropy(targets=0.0)], 
        dis_loss_weights=[0.5, 0.5], 
        dis_metrics=[Mean(name="mean_real"), Mean(name="mean_fake")],
        dis_decoders=None,
        dis_collate_fn=CollateList(inputs_axis=[0, 1], targets_axis=[]),

        gen_optimizer="Adam",
        gen_losses=[BinaryCrossEntropy(targets=1.0), MeanAbsoluteError()],
        gen_loss_weights=[1.0, 100.0], 
        gen_metrics=[Mean(name="mean_fake")],
        gen_decoders=None,
        gen_collate_fn=CollateList(inputs_axis=[1], targets_axis=[0])
    ):

        super(ConditionalGAN, self).compile(
            dis_optimizer=dis_optimizer,
            dis_losses=dis_losses,
            dis_loss_weights=dis_loss_weights,
            dis_metrics=dis_metrics,
            dis_decoders=dis_decoders,
            dis_collate_fn=dis_collate_fn,

            gen_optimizer=gen_optimizer,
            gen_losses=gen_losses,
            gen_loss_weights=gen_loss_weights,
            gen_metrics=gen_metrics,
            gen_decoders=gen_decoders,
            gen_collate_fn=gen_collate_fn
        )

    class DiscriminatorGAN(Model):

        def __init__(self, generator, discriminator):
            super().__init__()
            self.generator = generator
            self.discriminator = discriminator

        def forward(self, inputs):

            # Unpack Inputs
            real, label = inputs

            # Forward Generator
            self.generator.fake = self.generator(label)

            # Forward Discriminator
            pred_real = self.discriminator([real, label])
            pred_fake = self.discriminator([self.generator.fake.detach(), label])

            return {"real": pred_real, "fake": pred_fake}

    class GeneratorGAN(Model):

        def __init__(self, generator, discriminator):
            super().__init__()
            self.generator = generator
            self.discriminator = discriminator

        def forward(self, inputs):

            # Unpack Inputs
            label = inputs

            # Forward Discriminator
            preds = self.discriminator([self.generator.fake, label])

            return {"output": preds, "L1": self.generator.fake}

    def on_epoch_end(self, saving_period, callback_path, epoch, inputs, targets):
        super(GenerativeAdversarialNetwork, self).on_epoch_end(saving_period, callback_path, epoch, inputs, targets)

        # Eval Mode
        self.eval()

        # Batch Size
        nrow = 10

        # Slice Batch
        label = inputs["generator"]
        if nrow ** 2 > label.size(0):
            nrow = int(label.size(0) ** 0.5)
        label = label[:nrow ** 2]

        #label = torch.arange(10, dtype=torch.long, device=label.device).repeat(10)

        # Forward Generator
        with torch.no_grad():
            gen_outputs = self.generator(label)

        # Save Image Grid
        torchvision.utils.save_image(gen_outputs, callback_path + "samples_" + str(epoch + 1) + ".png", nrow=nrow, normalize=True)

class CycleGAN(GenerativeAdversarialNetwork):

    def __init__(self, G_AB, G_BA, D_A, D_B, name="Cycle Generative Adversarial Network"):
        super(CycleGAN, self).__init__(generator=[G_AB, G_BA], discriminator=[D_A, D_B], name=name)

    def summary(self, show_dict=False):

        print(self.name)
        print("Model Parameters:", self.gen_gan.num_params())
        
        print("G_AB Parameters:", self.num_params(self.gen_gan.G_AB))
        if show_dict:
            self.show_dict(self.gen_gan.G_AB)
            
        print("G_BA Parameters:", self.num_params(self.gen_gan.G_BA))
        if show_dict:
            self.show_dict(self.gen_gan.G_BA)

        print("D_A Parameters:", self.num_params(self.gen_gan.D_A))
        if show_dict:
            self.show_dict(self.gen_gan.D_A)

        print("D_B Parameters:", self.num_params(self.gen_gan.D_B))
        if show_dict:
            self.show_dict(self.gen_gan.D_B)

    def train_step(self, inputs, targets, mixed_precision, grad_scaler, accumulated_steps, acc_step):

        # Init Dict
        batch_losses = {}
        batch_metrics = {}

        # Discriminator Step
        self.set_require_grad(self.discriminator, False)
        dis_batch_losses, dis_batch_metrics, _ = self.gen_gan.train_step(inputs["generator"], targets["generator"], mixed_precision, grad_scaler, accumulated_steps, acc_step)
        batch_losses.update({"gen_" + key: value for key, value in dis_batch_losses.items()})
        batch_metrics.update({"gen_" + key: value for key, value in dis_batch_metrics.items()})

        # Generator Step
        self.set_require_grad(self.discriminator, True)
        gen_batch_losses, gen_batch_metrics, _ = self.dis_gan.train_step(inputs["discriminator"], targets["discriminator"], mixed_precision, grad_scaler, accumulated_steps, acc_step)
        batch_losses.update({"dis_" + key: value for key, value in gen_batch_losses.items()})
        batch_metrics.update({"dis_" + key: value for key, value in gen_batch_metrics.items()})

        return batch_losses, batch_metrics, _

    def compile(
        self,

        gen_optimizer="Adam",
        gen_losses=[MeanAbsoluteError(), MeanAbsoluteError(), MeanAbsoluteError(), MeanAbsoluteError(), BinaryCrossEntropy(targets=1.0), BinaryCrossEntropy(targets=1.0)],
        gen_loss_weights=[10.0, 10.0, 5.0, 5.0, 1.0, 1.0],
        gen_metrics=[None, None, None, None, BinaryAccuracy(targets=1.0), BinaryAccuracy(targets=1.0)],
        gen_decoders=[None, None, None, None, ThresholdDecoder(), ThresholdDecoder()],
        gen_collate_fn=CollateList(inputs_axis=[0, 1], targets_axis=[0, 1, 0, 1]),

        dis_optimizer="Adam", 
        dis_losses=[BinaryCrossEntropy(targets=1.0), BinaryCrossEntropy(targets=0.0), BinaryCrossEntropy(targets=1.0), BinaryCrossEntropy(targets=0.0)],
        dis_loss_weights=[0.5, 0.5, 0.5, 0.5], 
        dis_metrics=[BinaryAccuracy(targets=1.0, name="acc_A_real"), BinaryAccuracy(targets=0.0, name="acc_A_fake"), BinaryAccuracy(targets=1.0, name="acc_B_real"), BinaryAccuracy(targets=0.0, name="acc_B_fake")],
        dis_decoders=ThresholdDecoder(),
        dis_collate_fn=CollateList(inputs_axis=[0, 1], targets_axis=[])
    ):
        super(CycleGAN, self).compile(
            dis_optimizer=dis_optimizer,
            dis_losses=dis_losses,
            dis_loss_weights=dis_loss_weights,
            dis_metrics=dis_metrics,
            dis_decoders=dis_decoders,
            dis_collate_fn=dis_collate_fn,

            gen_optimizer=gen_optimizer,
            gen_losses=gen_losses,
            gen_loss_weights=gen_loss_weights,
            gen_metrics=gen_metrics,
            gen_decoders=gen_decoders,
            gen_collate_fn=gen_collate_fn
        )

    class GeneratorGAN(Model):

        def __init__(self, generator, discriminator):
            super().__init__()

            self.G_AB, self.G_BA = generator
            self.D_A, self.D_B = discriminator

        def forward(self, inputs):

            # unpack Inputs
            A, B = inputs

            # Cycle A
            self.G_AB.B_fake = self.G_AB(A)
            A_cycle = self.G_AB(self.G_AB.B_fake)

            # Cycle B
            self.G_BA.A_fake = self.G_BA(B)
            B_cycle = self.G_AB(self.G_BA.A_fake)

            # Identity A
            if True:#self.loss_weights["A_idt"] > 0:
                A_idt = self.G_BA(A)
            else:
                A_idt = None

            # Identity B
            if True:#self.loss_weights["B_idt"] > 0:
                B_idt = self.G_AB(B)
            else:
                B_idt = None

            # Dis A Fake
            A_pred = self.D_A(self.G_BA.A_fake)

            # Dis B Fake
            B_pred = self.D_B(self.G_AB.B_fake)

            return {"A_cycle": A_cycle, "B_cycle": B_cycle, "A_idt": A_idt, "B_idt": B_idt, "A_pred": A_pred, "B_pred": B_pred}

    class DiscriminatorGAN(Model):

        def __init__(self, generator, discriminator):
            super().__init__()

            self.G_AB, self.G_BA = generator
            self.D_A, self.D_B = discriminator

        def forward(self, inputs):

            # unpack Inputs
            A, B = inputs

            # D_A pred
            A_pred_real = self.D_A(A)
            A_pred_fake = self.D_A(self.G_BA.A_fake.detach())

            # D_B pred
            B_pred_real = self.D_B(B)
            B_pred_fake = self.D_B(self.G_AB.B_fake.detach())

            return {"A_pred_real": A_pred_real, "A_pred_fake": A_pred_fake, "B_pred_real": B_pred_real, "B_pred_fake": B_pred_fake}

    def on_epoch_end(self, saving_period, callback_path, epoch, inputs, targets):
        super(CycleGAN, self).on_epoch_end(saving_period, callback_path, epoch, inputs, targets)

###############################################################################
# Auto Encoder Models
###############################################################################

class AutoEncoder(Model):

    def __init__(self, encoder, decoder, name="Auto Encoder"):
        super(AutoEncoder, self).__init__(name=name)

        # Encoder
        self.encoder = encoder

        # Decoder
        self.decoder = decoder

    def summary(self, show_dict=False):

        print(self.name)
        print("Model Parameters: {:,}".format(self.num_params()))

        print("Encoder Parameters: {:,}".format(self.num_params(self.encoder)))
        if show_dict:
            self.show_dict(self.encoder)
            
        print("Decoder Parameters: {:,}".format(self.num_params(self.decoder)))
        if show_dict:
            self.show_dict(self.decoder)
            
    def compile(
        self, 
        losses=MeanAbsoluteError(),
        loss_weights=None,
        optimizer="Adam",
        metrics=None,
        decoders=None,
        collate_fn=CollateList(inputs_axis=[0], targets_axis=[0])
    ):

        super(AutoEncoder, self).compile(
            losses=losses,
            loss_weights=loss_weights,
            optimizer=optimizer,
            metrics=metrics,
            decoders=decoders,
            collate_fn=collate_fn
        )
    
    def forward(self, inputs):

        # Unpack Inputs
        x = inputs

        # Encode
        x = self.encoder(x)

        # Decode
        x = self.decoder(x)

        return x

class VariationalAutoEncoder(AutoEncoder):

    """ Variational Auto Encoder

    KLD weight Beta = 1 by default

    References: 
        Auto-Encoding Variational Bayes, Kingma et al.
        https://arxiv.org/abs/1312.6114
    
    """

    def __init__(self, encoder, decoder, name="Variational Auto Encoder"):
        super(VariationalAutoEncoder, self).__init__(encoder=encoder, decoder=decoder, name=name)

    def compile(
        self, 
        losses=[BinaryCrossEntropy(reduction="mean_batch"), KullbackLeiblerDivergence()],
        loss_weights=[1.0, 1.0],
        optimizer="Adam",
        metrics=None,
        decoders=None,
        collate_fn=CollateList(inputs_axis=[0], targets_axis=[0])
    ):

        super(AutoEncoder, self).compile(
            losses=losses,
            loss_weights=loss_weights,
            optimizer=optimizer,
            metrics=metrics,
            decoders=decoders,
            collate_fn=collate_fn
        )

    def forward(self, inputs):

        # Unpack Inputs
        x = inputs

        # Encode
        mean, log_var = self.encoder(x)

        # Sample Noise
        noise = torch.randn(size=mean.size(), device=mean.device) 
        
        # Reparameterize
        x_latent = noise * torch.exp(log_var * 0.5) + mean

        # Decode
        x = self.decoder(x_latent)

        return {"output": x, "KLD": [mean, log_var]}

    def on_epoch_end(self, saving_period, callback_path, epoch, inputs, targets):
        super(AutoEncoder, self).on_epoch_end(saving_period, callback_path, epoch, inputs, targets)

        # Eval Mode
        self.eval()

        # Number of Rows
        nrow = 10

        # Eval Mode
        self.eval()

        # Forward VAE
        with torch.no_grad():

            # Encode
            mean, log_var = self.encoder(inputs[:nrow])
            
            # Reparameterize
            x_latent = torch.randn(size=mean.size(), device=mean.device)  * torch.exp(log_var * 0.5) + mean

            # Decode
            outputs = self.decoder(x_latent)

        # Plot
        plt.figure(figsize=(15, 15))
        for b in range(nrow):

            plt.subplot(nrow, 2, 2*b+1)
            plt.title("Input")
            plt.imshow(inputs[b].permute(1, 2, 0).cpu())

            plt.subplot(nrow, 2, 2*b+2)
            plt.title("Outputs")
            plt.imshow(outputs[b].permute(1, 2, 0).cpu())

        plt.savefig(callback_path + "samples_" + str(epoch + 1) + ".png")
        plt.close()

        # Number of Rows
        nrow = 30

        # Generate 2D grid
        try:
            with torch.no_grad():
                
                # Reparameterize
                x = torch.linspace(-5, 5, steps=nrow).repeat(nrow)
                x = torch.stack([x, x.reshape(nrow, nrow).transpose(0, 1).flatten()], dim=-1)

                # Decode
                outputs = self.decoder(x)

            # Save Image Grid
            torchvision.utils.save_image(outputs, callback_path + "samples_gen_" + str(epoch + 1) + ".png", nrow=nrow, normalize=True, )
        except:
            pass

        
class VectorQuantizedVAE(AutoEncoder):

    """ Vector Quantized Variational Auto Encoder

    Reference: "Neural Discrete Representation Learning", van den Oord et al.
    https://arxiv.org/abs/1711.00937
    
    """

    def __init__(self, encoder, decoder, num_embeddings, embedding_dim, name="Vector Quantized Variational Auto Encoder"):
        super(VectorQuantizedVAE, self).__init__(encoder=encoder, decoder=decoder, name=name)

        # Quantization Module
        self.quantizer = VectorQuantizer(num_embeddings=num_embeddings, embedding_dim=embedding_dim)

    def summary(self, show_dict=False):
        super(VectorQuantizedVAE, self).summary(show_dict=show_dict)

        print("Codebook Parameters: {:,}".format(self.num_params(self.quantizer)))
        if show_dict:
            self.show_dict(self.quantizer)

    def compile(
        self, 
        losses=MeanSquaredError(),
        loss_weights=None,
        optimizer="Adam",
        metrics=None,
        decoders=None,
        collate_fn=CollateList(inputs_axis=[0], targets_axis=[0])
    ):

        super(AutoEncoder, self).compile(
            losses=losses,
            loss_weights=loss_weights,
            optimizer=optimizer,
            metrics=metrics,
            decoders=decoders,
            collate_fn=collate_fn
        )

    def forward(self, inputs):

        # Encode
        x = self.encoder(inputs)

        # Quantize
        x_q = self.quantizer(x)

        # Decode
        x = self.decoder(x_q)

        return x

    def encode(self, inputs):

        with torch.no_grad():

            # Unpack Inputs
            x = inputs

            # Encode
            x = self.encoder(x)

            # Encode to indices
            indices = self.quantizer.encode(x)

        return indices

    def decode(self, indices):

        with torch.no_grad():

            # Decode indices
            x_q = self.quantizer.decode(indices)

            # Decode
            x = self.decoder(x_q)

        return x

    def forward_generate(self, inputs):

        return self.encode(inputs)

    def on_epoch_end(self, saving_period, callback_path, epoch, inputs, targets, writer):
        super(VectorQuantizedVAE, self).on_epoch_end(saving_period, callback_path, epoch, inputs, targets, writer)

        # Eval Mode
        self.eval()

        # Forward VAE
        with torch.no_grad():
            indices = self.encode(inputs)
            rec = self.decode(indices)
        
        B = 10
        fig = plt.figure(figsize=(15, 15))

        for b in range(B):

            plt.subplot(B, 3, 3*b+1)
            if inputs.dim() == 2: # (B, T)
                plt.title("Input")
                plt.plot(inputs[b].cpu())
            elif inputs.dim() == 5: # (B, C, T, H, W)
                t = torch.randint(0, inputs.size(2), size=())
                plt.title("Input, t = {}".format(t))
                plt.imshow(inputs[b, :, t].permute(1, 2, 0).cpu())
            else: # (B, C, H, W)
                plt.title("Input")
                plt.imshow(inputs[b].permute(1, 2, 0).cpu())

            plt.subplot(B, 3, 3*b+2)
            if indices.dim() == 2: # (B, T)
                plt.title("Encoded")
                plt.plot(indices[b].cpu())
            elif indices.dim() == 4: # (B, T, H, W)
                t_ind = int(t * indices.size(1) / inputs.size(2))
                plt.title("Encoded, t_enc = {}".format(t_ind))
                plt.imshow(indices[b, t_ind].cpu())
            else: # (B, H, W)
                plt.title("Encoded")
                plt.imshow(indices[b].cpu())

            plt.subplot(B, 3, 3*b+3)
            if rec.dim() == 2: # (B, T)
                plt.title("Decoded")
                plt.plot(rec[b].cpu())
            elif rec.dim() == 5: # (B, C, T, H, W)
                plt.title("Decoded, t = {}".format(t))
                plt.imshow(rec[b, :, t].permute(1, 2, 0).cpu())
            else: # (B, C, H, W)
                plt.title("Decoded")
                plt.imshow(rec[b].permute(1, 2, 0).cpu())

        # Add Figure to logs
        writer.add_figure("Samples/", fig, epoch + 1)

class VectorQuantizedEMAVAE(VectorQuantizedVAE):

    def __init__(self, encoder, decoder, num_embeddings, embedding_dim, gamma=0.99, name="Exp Moving Vector Quantized Variational Auto Encoder"):
        super(VectorQuantizedVAE, self).__init__(encoder=encoder, decoder=decoder, name=name)

        # Quantization Module
        self.quantizer = VectorQuantizerEMA(num_embeddings=num_embeddings, embedding_dim=embedding_dim, gamma=gamma)

class GumbelVectorQuantizedVAE(VectorQuantizedVAE):

    def __init__(self, encoder, decoder, num_embeddings, embedding_dim, name="Gumbel Softmax Vector Quantized Variational Auto Encoder"):
        super(VectorQuantizedVAE, self).__init__(encoder=encoder, decoder=decoder, name=name)

        # Quantization Module
        self.quantizer = GumbelSoftmaxQuantizer(num_embeddings=num_embeddings, embedding_dim=embedding_dim)

class VectorQuantizedGAN(GenerativeAdversarialNetwork):

    def __init__(self, encoder, decoder, num_embeddings, embedding_dim, discriminator, name="Vector Quantized Generative Adversarial Network"):
        super(VectorQuantizedGAN, self).__init__(generator=[encoder, decoder, VectorQuantizer(num_embeddings, embedding_dim)], discriminator=discriminator, name=name)

    def summary(self, show_dict=False):

        print(self.name)
        print("Model Parameters:", self.gen_gan.num_params())
        
        print("Encoder Parameters:", self.num_params(self.gen_gan.encoder))
        if show_dict:
            self.show_dict(self.gen_gan.encoder)
            
        print("Decoder Parameters:", self.num_params(self.gen_gan.decoder))
        if show_dict:
            self.show_dict(self.gen_gan.decoder)

        print("Codebook Parameters:", self.num_params(self.gen_gan.quantizer))
        if show_dict:
            self.show_dict(self.gen_gan.quantizer)

        print("Disriminator Parameters:", self.num_params(self.gen_gan.discriminator))
        if show_dict:
            self.show_dict(self.gen_gan.discriminator)

    def compile(
        self, 
        dis_optimizer="Adam", 
        dis_losses=[BinaryCrossEntropy(targets=1.0), BinaryCrossEntropy(targets=0.0)], 
        dis_loss_weights=[0.5, 0.5], 
        dis_metrics=[Mean(name="mean_real"), Mean(name="mean_fake")],
        dis_decoders=None,
        dis_collate_fn=CollateList(inputs_axis=[0], targets_axis=[]),

        gen_optimizer="Adam",
        gen_losses=[BinaryCrossEntropy(targets=1.0), MeanSquaredError(), MeanSquaredError(), MeanSquaredError()],
        gen_loss_weights=[1.0, 1.0, 1.0, 0.25], 
        gen_metrics=[Mean(name="mean_fake")],
        gen_decoders=None,
        gen_collate_fn=CollateList(inputs_axis=[], targets_axis=[0])
    ):

        super(VectorQuantizedGAN, self).compile(
            dis_optimizer=dis_optimizer,
            dis_losses=dis_losses,
            dis_loss_weights=dis_loss_weights,
            dis_metrics=dis_metrics,
            dis_decoders=dis_decoders,
            dis_collate_fn=dis_collate_fn,

            gen_optimizer=gen_optimizer,
            gen_losses=gen_losses,
            gen_loss_weights=gen_loss_weights,
            gen_metrics=gen_metrics,
            gen_decoders=gen_decoders,
            gen_collate_fn=gen_collate_fn
        )

    def encode(self, inputs):

        with torch.no_grad():

            # Unpack Inputs
            x = inputs

            # Encode
            x = self.encoder(x)

            # Encode to indices
            indices = self.quantizer.encode(x)

        return indices

    def decode(self, indices):

        with torch.no_grad():

            # Decode indices
            x_q = self.quantizer.decode(indices)

            # Decode
            x = self.decoder(x_q)

        return x

    class DiscriminatorGAN(Model):

        def __init__(self, generator, discriminator):
            super().__init__()

            self.encoder, self.decoder, self.quantizer = generator
            self.discriminator = discriminator

        def forward(self, inputs):

            # Encode
            x = self.encoder(inputs)

            # Quantize
            self.quantizer.outputs = self.quantizer(x)

            # Decode
            self.decoder.fake = self.decoder(self.quantizer.outputs[0])

            # Forward Discriminator
            pred_real = self.discriminator(inputs)
            pred_fake = self.discriminator(self.decoder.fake.detach())

            return {"real": pred_real, "fake": pred_fake}

    class GeneratorGAN(Model):

        def __init__(self, generator, discriminator):
            super().__init__()
            
            self.encoder, self.decoder, self.quantizer = generator
            self.discriminator = discriminator

        def forward(self, inputs):

            # Unpack Quantizer Outputs
            x_q, output_codebook, target_codebook, output_commit, target_commit, div = self.quantizer.outputs

            # Forward Discriminator
            preds = self.discriminator(self.decoder.fake)

            # Update Targets
            self.additional_targets["output_codebook"] = target_codebook
            self.additional_targets["output_commit"] = target_commit

            # Infos
            self.infos["codebook_div"] = round(div, 2)

            return {"output_dis": preds, "output_rec": self.decoder.fake, "output_codebook": output_codebook, "output_commit": output_commit}

    def on_epoch_end(self, saving_period, callback_path, epoch, inputs, targets):
        super(GenerativeAdversarialNetwork, self).on_epoch_end(saving_period, callback_path, epoch, inputs, targets)

        # Eval Mode
        self.eval()

        # Batch Size
        nrow = 10
        B = nrow**2

        # Eval Mode
        self.eval()

        inputs = inputs["discriminator"]

        # Forward VAE
        with torch.no_grad():
            indices = self.encode(inputs)
            rec = self.decode(indices)
        
        B = 10
        plt.figure(figsize=(15, 15))

        for b in range(B):

            plt.subplot(B, 3, 3*b+1)
            plt.title("Input")
            if len(inputs.size()) == 2:
                plt.plot(inputs["inputs"][b].cpu())
            elif len(inputs.size()) == 5:
                plt.imshow(inputs[b, 0, 0].cpu())
            else:
                plt.imshow(inputs[b].permute(1, 2, 0).cpu() * 0.5 + 0.5)

            plt.subplot(B, 3, 3*b+2)
            plt.title("Encoded")
            if len(indices.size()) == 1:
                plt.plot(indices[b].cpu())
            elif len(indices.size()) == 4:
                plt.imshow(indices[b, 0].cpu())
            else:
                plt.imshow(indices[b].cpu())

            plt.subplot(B, 3, 3*b+3)
            plt.title("Decoded")
            if len(rec.size()) == 2:
                plt.plot(rec[b].cpu())
            elif len(rec.size()) == 5:
                plt.imshow(rec[b, 0, 0].cpu())
            else:
                plt.imshow(rec[b].permute(1, 2, 0).cpu() * 0.5 + 0.5)

        plt.savefig(callback_path + "samples_" + str(epoch + 1) + ".png")
        plt.close()

###############################################################################
# Reinforcement Learning Models
###############################################################################

class MonteCarloPolicyGradient(Model):

    def __init__(self, net, env, batch_size=1000, max_steps=10000, gamma=0.99, discrete=True, name="Monte-Carlo Policy Gradient Model"):
        super(MonteCarloPolicyGradient, self).__init__(name=name)

        # Init
        self.net = net
        self.env = env
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.gamma = gamma
        self.infos["episodes"] = 0
        self.discrete = discrete

    def compile(
        self, 
        losses=[NegativeLogLikelihood()],
        loss_weights=None,
        optimizer="Adam",
        metrics=[None, Mean(name="mean_reward")],
        decoders=None,
        collate_fn=CollateList(inputs_axis=[], targets_axis=[])
    ):

        super(MonteCarloPolicyGradient, self).compile(
            losses=losses,
            loss_weights=loss_weights,
            optimizer=optimizer,
            metrics=metrics,
            decoders=decoders,
            collate_fn=collate_fn
        )

    def forward(self, inputs):

        # Init
        returns = []
        action_probs = []
        total_rewards = []
        b = 0

        while b < self.batch_size:

            # Reset Env
            state = self.env.reset().to(self.device)
            ep_action_probs = []
            ep_rewards = []

            # Episode loop
            for step in range(self.max_steps):

                # Forward Model
                pred_action = self.net(state.unsqueeze(dim=0))

                # Sample Action
                if self.discrete:
                    action = pred_action.multinomial(num_samples=1).item()
                else:
                    noise = torch.randn(1, 1, device=self.device)
                    action = (pred_action[0] + pred_action[1] * noise).item()

                # Forward Env
                state, reward, done = self.env.step(action)
                state = state.to(self.device)

                # Action Prob
                if self.discrete:
                    ep_action_probs.append(pred_action[0, action])
                else:
                    ep_action_probs.append(1 / (pred_action[1][0] * (2 * torch.pi) ** 0.5) * (- 0.5 * ((action - pred_action[0][0]) / pred_action[1][0]) ** 2).exp())

                # Update
                ep_rewards.append(reward)
                b += 1

                # Done
                if done:
                    self.infos["episodes"] += 1
                    break

            # Compute Episode Returns
            ep_returns = []
            rewards_sum = 0
            for reward in reversed(ep_rewards):
                rewards_sum = reward + self.gamma * rewards_sum
                ep_returns.insert(0, rewards_sum)
            returns.append(torch.tensor(ep_returns, device=self.device))

            # Append Epidode Action Probs
            action_probs.append(torch.stack(ep_action_probs, dim=0))

            # Compute Episode Total Rewards
            total_rewards.append(torch.tensor(ep_rewards, device=self.device).sum())

        # Concat Episodes Histories
        returns = torch.concat(returns, dim=0)
        action_probs = torch.concat(action_probs, dim=0)
        total_rewards = torch.stack(total_rewards, dim=0)

        # Update Targets
        self.additional_targets["actor"] = returns

        self.infos["batch_rewards"] = "{:.2f}".format(total_rewards.mean())

        return {"actor": action_probs, "rewards": total_rewards}

class MonteCarloActorCritic(Model):

    def __init__(self, net, env, batch_size=1000, max_steps=10000, gamma=0.99, name="Monte-Carlo Actor Critic Model"):
        super(MonteCarloActorCritic, self).__init__(name=name)

        # Init
        self.net = net
        self.env = env
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.gamma = gamma

    def compile(
        self, 
        losses=[NegativeLogLikelihood(), HuberLoss()],
        loss_weights=[1, 1],
        optimizer="Adam",
        metrics=[None, None, Mean(name="mean_reward")],
        decoders=None,
        collate_fn=CollateList(inputs_axis=[], targets_axis=[])
    ):

        super(MonteCarloActorCritic, self).compile(
            losses=losses,
            loss_weights=loss_weights,
            optimizer=optimizer,
            metrics=metrics,
            decoders=decoders,
            collate_fn=collate_fn
        )

    def forward(self, inputs):

        # Init
        returns = []
        action_probs = []
        values = []
        total_rewards = []
        b = 0

        while b < self.batch_size:

            # Reset Env
            state = self.env.reset()
            ep_action_probs = []
            ep_values = []
            ep_rewards = []

            # Episode loop
            for step in range(self.max_steps):

                # Forward Model
                pred_action, pred_value = self.net(torch.tensor(state, device=self.device).unsqueeze(dim=0))

                # Sample Action
                action = pred_action.multinomial(num_samples=1).item()

                # Forward Env
                state, reward, done, info = self.env.step(action)

                # Update Histories
                ep_action_probs.append(pred_action[0, action])
                ep_values.append(pred_value[0, 0])
                ep_rewards.append(reward)
                b += 1

                # Done
                if done:
                    break

            # Compute Episode Returns
            ep_returns = []
            rewards_sum = 0
            for reward in reversed(ep_rewards):
                rewards_sum = reward + self.gamma * rewards_sum
                ep_returns.insert(0, rewards_sum)
            returns.append(torch.tensor(ep_returns, device=self.device))

            # Append Epidode Action Probs
            action_probs.append(torch.stack(ep_action_probs, dim=0))

            # Append Epidode Values
            values.append(torch.stack(ep_values, dim=0))

            # Compute Episode Total Rewards
            total_rewards.append(torch.tensor(ep_rewards, device=self.device).sum())

        # Concat Episodes Histories
        returns = torch.concat(returns, dim=0)
        action_probs = torch.concat(action_probs, dim=0)
        values = torch.concat(values, dim=0)
        total_rewards = torch.stack(total_rewards, dim=0)

        # Compute Advantages
        advantages = returns - values.detach()
        #advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-9)

        # Update Targets
        self.additional_targets["actor"] = advantages
        self.additional_targets["critic"] = returns

        return {"actor": action_probs, "critic": values, "reward": total_rewards}

class QValueNetwork(Model):

    """Off Policy Deep Q Learning with experience replay

    References:
        Q-Learning, Watkins et al.
        Playing Atari with Deep Reinforcement Learning, Mnih et al.

    """

    def __init__(self, net, env, batch_size=32, max_steps=10000, gamma=0.99, eps_min=0.1, eps_max=1.0, eps_random=50000, eps_steps=1000000, max_memory_length=100000, update_target_network_period=10000, update_period=4, name="Q Value Network"):
        super(QValueNetwork, self).__init__(name=name)

        # Networks
        self.net = net
        self.net_target = type(self.net)()
        self.net_target.load_state_dict(self.net.state_dict())

        # Env
        self.env = env

        # Training Params
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.gamma = gamma
        self.eps_min = eps_min
        self.eps = eps_max
        self.eps_inter = eps_max - eps_min
        self.eps_random = eps_random
        self.eps_steps = eps_steps
        self.update_period = update_period
        self.update_target_network_period = update_target_network_period
        self.max_memory_length = max_memory_length

        # Training Infos
        self.updated = False
        self.done = False
        self.action_step = 0
        self.episodes = 0
        self.buffer = []
        self.running_rewards = 0.0

    def save(self, path, save_optimizer=True):

        # Save Buffer
        torch.save({
            "buffer": self.buffer,
            "eps": self.eps,
            "action_step": self.action_step,
            "episodes": self.episodes,
            "running_rewards": self.running_rewards
            }, "/".join(path.split("/")[:-1]) + "/buffer" + str(int("".join(filter(str.isdigit, path.split("/")[-1]))) % 2) + ".ckpt")

        # save
        super(QValueNetwork, self).save(path, save_optimizer)

    def load(self, path):

        # Load Buffer
        try:
            buffer_checkpoint = torch.load(path + "b", map_location=next(self.parameters()).device)
            #buffer_checkpoint = torch.load("/".join(path.split("/")[:-1]) + "/buffer.ckpt", map_location=next(self.parameters()).device)
            self.buffer = buffer_checkpoint["buffer"]
            self.eps = buffer_checkpoint["eps"]
            self.action_step = buffer_checkpoint["action_step"]
            self.episodes = buffer_checkpoint["episodes"]
            self.running_rewards = buffer_checkpoint["running_rewards"]
        except:
            print("Buffer not loaded...")

        # load
        super(QValueNetwork, self).load(path)

    def compile(
        self, 
        losses=[HuberLoss()],
        loss_weights=None,
        optimizer="Adam",
        metrics=None,
        decoders=None,
        collate_fn=CollateList(inputs_axis=[], targets_axis=[])
    ):

        super(QValueNetwork, self).compile(
            optimizer=Adam(params=self.net.parameters(), lr=0.00025) if optimizer == "Adam" else optimizer,
            losses=losses,
            loss_weights=loss_weights,
            metrics=metrics,
            decoders=decoders,
            collate_fn=collate_fn
        )

    def summary(self, show_dict=False):

        print(self.name)
        print("Model Parameters:", self.num_params(self.net))
        if show_dict:
            self.show_dict(self.net)

    def forward(self, inputs):

        # Episode loop
        while True:

            # Reset Env
            if self.updated:
                self.updated = False
            else:
                self.state = self.env.reset().to(self.device)
                self.ep_total_rewards = 0.0

            # Episode loop
            for step in range(self.max_steps):

                # Update target Network
                if self.action_step % self.update_target_network_period == 0:
                    self.net_target.load_state_dict(self.net.state_dict())

                # Limit the state and reward history
                if len(self.buffer) > self.max_memory_length:
                    del self.buffer[:1]

                # Done 
                if self.done:
                    self.episodes += 1
                    self.done = False
                    self.running_rewards = 0.05 * self.ep_total_rewards + (1 - 0.05) * self.running_rewards
                    break

                # Exploration Step
                if self.action_step < self.eps_random or self.eps > torch.rand(1).item():
                    action = self.env.sample()
                else:
                    with torch.no_grad():
                        action = self.net(self.state.unsqueeze(dim=0)).argmax(dim=-1).item()

                # Update Step
                self.action_step += 1

                # Decay probability of taking random action
                self.eps -= self.eps_inter / self.eps_steps
                self.eps = max(self.eps, self.eps_min)

                # Forward Env
                state_next, reward, self.done = self.env.step(action)
                state_next = state_next.to(self.device)

                # Update Buffer
                self.buffer.append((self.state, state_next, torch.tensor(action, device=self.device), torch.tensor(self.done, device=self.device, dtype=torch.float32), torch.tensor(reward, device=self.device)))
                self.ep_total_rewards += reward
                self.state = state_next

                # Update Network
                if self.action_step % self.update_period == 0 and len(self.buffer) > self.batch_size:

                    # Sample Batch from Buffer
                    samples = [self.buffer[i] for i in torch.randint(low=0, high=len(self.buffer), size=(self.batch_size,))]
                    sample_states = torch.stack([sample[0] for sample in samples])
                    sample_states_next = torch.stack([sample[1] for sample in samples])
                    sample_actions = torch.stack([sample[2] for sample in samples])
                    sample_dones = torch.stack([sample[3] for sample in samples])
                    sample_rewards = torch.stack([sample[4] for sample in samples])

                    # Predict Future Returns
                    with torch.no_grad():
                        future_returns = self.net_target(sample_states_next)

                    # Compute Targets Returns
                    returns_targets = sample_rewards + self.gamma * future_returns.max(dim=-1)[0]

                    # If final step set the last value to -1
                    returns_targets = returns_targets * (1 - sample_dones) - sample_dones

                    # Create a mask so we only calculate loss on the updated Q-values
                    masks = F.one_hot(sample_actions, num_classes=self.env.actions)

                    # Forward Model
                    returns = (self.net(sample_states) * masks).sum(dim=-1)

                    # Add Targets
                    self.additional_targets["q_value"] = returns_targets

                    # Set Updated to True
                    self.updated = True

                    # Update Infos
                    self.infos["action_step"] = self.action_step
                    self.infos["episodes"] = self.episodes
                    self.infos["running_rewards"] = "{:.2f}".format(self.running_rewards)
                    self.infos["eps"] = "{:.2e}".format(self.eps)
                    self.infos["ep_rewards"] = self.ep_total_rewards

                    return {"q_value": returns}

    def play(self, verbose=False, eps=0.05):

        # Reset
        state = self.env.reset().to(self.device)
        total_rewards = 0
        total_value = 0
        step = 0

        # Episode loop
        while 1:

            # Sample Action
            values = self.net(state.unsqueeze(dim=0))
            action = values.argmax(dim=-1).item()
            value = values.max()

            # Random Action
            if eps > torch.rand(1).item():
                action = self.env.sample()

            # Forward Env
            state, reward, done = self.env.step(action)
            total_value += value
            step += 1

            # Verbose
            if verbose:
                infos = "step {}, action {}, reward {}, done {}, total {}, Q {:.2f}, mean Q {:.2f}".format(step, action, reward, done, total_rewards, value, total_value / step)
                print(infos)
                plt.title(infos)
                plt.imshow(state[0])
                plt.pause(0.001)
                plt.close()

            # Update total_reward
            total_rewards += reward

            # Break
            if done:
                break

        return total_rewards, step, total_value / step

    def eval_step(self, inputs, targets, verbose=False):

        with torch.no_grad():
            score, steps, value = self.play(verbose=verbose)

        # Update Infos
        self.infos["ep_score"] = score
        self.infos["ep_steps"] = steps
        self.infos["ep_Q"] = "{:.4f}".format(value)

        return {}, {"score": score, "steps": steps, "Q": value}, {}, {}

class DeepDeterministicPolicyGradient(Model):

    """Deep Deterministic Policy Gradient (DDPG)

    Model-Free Off-Policy Continuous Actor-Critic algorithm with Experience Replay

    References:
        Continuous Control With Deep Reinforcement Learning, Lillicrap et al.
    
    """

    def __init__(self, p_net, q_net, env, batch_size=64, gamma=0.99, tau=0.001, noise_theta=0.15, noise_std=0.2, noise_dt=0.05, buffer_size=50000, update_period=1, reward_done=None, include_done_transition=True, name="Deep Deterministic Policy Gradient Model"):
        super(DeepDeterministicPolicyGradient, self).__init__(name=name)

        # Policy Networks
        self.p_net = p_net
        self.p_target = type(self.p_net)()
        self.p_target.load_state_dict(self.p_net.state_dict())
        self.set_require_grad(self.p_target, False)


        # Q-Value Networks
        self.q_net = q_net
        self.q_target = type(self.q_net)()
        self.q_target.load_state_dict(self.q_net.state_dict())
        self.set_require_grad(self.q_target, False)

        # Critic
        self.critic = self.Critic(q_net=self.q_net, p_target=self.p_target, q_target=self.q_target, gamma=gamma, reward_done=reward_done)

        # Actor
        self.actor = self.Actor(p_net=self.p_net, q_net=self.q_net)

        # Env
        self.env = env
        self.state = self.env.reset()
        self.clip_low = self.env.env.action_space.low[0]
        self.clip_high = self.env.env.action_space.high[0]

        # Training Params
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.buffer_size = buffer_size
        self.noise = OrnsteinUhlenbeckProcess(num_actions=self.env.num_actions, mean=0, std=noise_std, theta=noise_theta, dt=noise_dt)
        self.update_period = update_period
        self.include_done_transition = include_done_transition

        # Training Infos
        self.episodes = 0
        self.buffer = []
        self.running_rewards = 0.0
        self.ep_rewards = 0.0
        self.action_step = 0

    def compile(
        self, 

        critic_optimizer="Adam",
        critic_losses=MeanSquaredError(),
        critic_loss_weights=None,
        critic_metrics=None,
        critic_decoders=None,
        critic_collate_fn=None,

        actor_optimizer="Adam",
        actor_losses=MeanLoss(targets_as_sign=False),
        actor_loss_weights=None,
        actor_metrics=None,
        actor_decoders=None,
        actor_collate_fn=None
    ):
        # Compile Critic
        self.critic.compile(
            optimizer=Adam(params=self.q_net.parameters(), lr=0.001, weight_decay=0.01) if critic_optimizer == "Adam" else critic_optimizer,
            losses=critic_losses,
            loss_weights=critic_loss_weights,
            metrics=critic_metrics,
            decoders=critic_decoders,
            collate_fn=critic_collate_fn
        )

        # Compile Actor
        self.actor.compile(
            optimizer=Adam(params=self.p_net.parameters(), lr=0.0001) if actor_optimizer == "Adam" else actor_optimizer,
            losses=actor_losses,
            loss_weights=actor_loss_weights,
            metrics=actor_metrics,
            decoders=actor_decoders,
            collate_fn=actor_collate_fn
        )

        # Model Step
        self.model_step = self.actor.optimizer.scheduler.model_step

        # Collate Function
        self.collate_fn = CollateList(inputs_axis=[], targets_axis=[])

        # Optimizer
        self.optimizer = {"critic": self.critic.optimizer, "actor": self.actor.optimizer}

        # Set Compiled to True
        self.compiled = True

    def summary(self, show_dict=False):

        print(self.name)
        print("Model Parameters:", self.num_params(self.p_net) + self.num_params(self.q_net))

        print("Actor Parameters:", self.num_params(self.p_net))
        if show_dict:
            self.show_dict(self.p_net)

        print("Critic Parameters:", self.num_params(self.q_net))
        if show_dict:
            self.show_dict(self.q_net)

    def env_step(self):

        # Get State
        state = self.state.to(self.device)

        # Forward Policy Network
        with torch.no_grad():
            action = self.p_net(state.unsqueeze(dim=0))

        # Action Info
        self.infos["actions"] = [round(a, 2) for a in action.squeeze(dim=0).tolist()]

        # Update Step
        self.action_step += 1

        # Add Noise
        action += self.noise()

        # Clip Action
        action = action.clip(self.clip_low, self.clip_high)

        # Env Step
        state_next, reward, done = self.env.step(action.squeeze(dim=0))

        # Render
        #self.env.env.render()

        # Limit the state and reward history
        if len(self.buffer) >= self.buffer_size:
            del self.buffer[:1]

        # Store Transitions
        if not done or self.include_done_transition:
            self.buffer.append((state, action.squeeze(dim=0), torch.tensor(reward, device=self.device, dtype=torch.float32), state_next.to(self.device), torch.tensor(done, device=self.device, dtype=torch.float32)))

        # Update ep rewards
        self.ep_rewards += reward

        # Done
        if done:
            self.state = self.env.reset()
            self.episodes += 1
            self.running_rewards = 0.05 * self.ep_rewards + (1 - 0.05) * self.running_rewards
            self.ep_rewards = 0.0
            self.noise.reset()
        else:
            self.state = state_next

    def sample_batch(self):

        # Sample Batch from Buffer
        samples = [self.buffer[i] for i in torch.randint(low=0, high=len(self.buffer), size=(self.batch_size,))]
        states = torch.stack([sample[0] for sample in samples])
        actions = torch.stack([sample[1] for sample in samples])
        rewards = torch.stack([sample[2] for sample in samples])
        states_next = torch.stack([sample[3] for sample in samples])
        dones = torch.stack([sample[4] for sample in samples])

        return [states, actions, rewards, states_next, dones]

    def train_step(self, inputs, targets, mixed_precision, grad_scaler, accumulated_steps, acc_step):

        # Environment Step
        env_step = 0
        while env_step < self.update_period or len(self.buffer) < self.batch_size:
          self.env_step()
          env_step += 1

        # Sample Inputs
        critic_inputs = self.sample_batch()
        actor_inputs = critic_inputs[0]

        # Init Dict
        batch_losses = {}
        batch_metrics = {}

        # Critic Step
        self.set_require_grad(self.q_net, True)
        critic_batch_losses, critic_batch_metrics, _ = self.critic.train_step(critic_inputs, targets, mixed_precision, grad_scaler, accumulated_steps, acc_step)
        batch_losses.update({"critic_" + key: value for key, value in critic_batch_losses.items()})
        batch_metrics.update({"critic_" + key: value for key, value in critic_batch_metrics.items()})

        # Actor Step
        self.set_require_grad(self.q_net, False)
        actor_batch_losses, actor_batch_metrics, _ = self.actor.train_step(actor_inputs, targets, mixed_precision, grad_scaler, accumulated_steps, acc_step)
        batch_losses.update({"actor_" + key: value for key, value in actor_batch_losses.items()})
        batch_metrics.update({"actor_" + key: value for key, value in actor_batch_metrics.items()})

        # Update Target Networks
        for param_target, param_net in zip(self.q_target.parameters(), self.q_net.parameters()):
            param_target.mul_(1 - self.tau)
            param_target.add_(self.tau * param_net.detach())
        for param_target, param_net in zip(self.p_target.parameters(), self.p_net.parameters()):
            param_target.mul_(1 - self.tau)
            param_target.add_(self.tau * param_net.detach())

        # Update Infos
        self.infos["episodes"] = self.episodes
        self.infos["running_rewards"] = round(self.running_rewards, 2)
        self.infos["ep_rewards"] = round(self.ep_rewards, 2)
        self.infos["step"] = self.model_step
        self.infos["action_step"] = self.action_step
        self.infos["critic_lr"] = self.critic.optimizer.param_groups[0]["lr"]
        self.infos["actor_lr"] = self.actor.optimizer.param_groups[0]["lr"]

        return batch_losses, batch_metrics, _

    class Critic(Model):

        def __init__(self, q_net, p_target, q_target, gamma, reward_done):
            super().__init__(name="Critic Model")

            self.q_net = q_net
            self.p_target = p_target
            self.q_target = q_target
            self.gamma = gamma
            self.reward_done = reward_done

        def forward(self, inputs):
            
            # Unpack Inputs
            states, actions, rewards, states_next, dones = inputs

            # Forward Q-Value network
            pred_returns = self.q_net([states, actions])

            # Compute Targets
            with torch.no_grad():
                self.additional_targets["value"] = rewards.unsqueeze(-1) + self.gamma * self.q_target([states_next, self.p_target(states_next)])

                if self.reward_done != None:
                    self.additional_targets["value"] = (rewards.unsqueeze(-1) + self.gamma * self.q_target([states_next, self.p_target(states_next)])) * (1 - dones.unsqueeze(-1)) + self.reward_done * dones.unsqueeze(-1)
                else:
                    self.additional_targets["value"] = rewards.unsqueeze(-1) + self.gamma * self.q_target([states_next, self.p_target(states_next)]) * (1 - dones.unsqueeze(-1)) 

            return {"value": pred_returns}
    
    class Actor(Model):

        def __init__(self, p_net, q_net):
            super().__init__(name="Actor Model")

            self.p_net = p_net
            self.q_net = q_net

        def forward(self, inputs):
            
            # Unpack Inputs
            states = inputs

            # Forward Policy Network
            pred_actions = self.p_net(states)

            # Forward Q-Value network
            pred_returns = self.q_net([states, pred_actions])

            return - pred_returns

    def play(self, verbose=False):

        # Reset
        state = self.env.reset().to(self.device)
        total_rewards = 0
        step = 0

        # Episode loop
        while 1:

            # Sample Action
            action = self.p_net(state.unsqueeze(dim=0))

            # Forward Env
            state, reward, done = self.env.step(action.squeeze(dim=0))
            state = state.to(self.device)
            step += 1
            total_rewards += reward

            # Verbose
            if verbose:
                infos = "step {}, action {}, reward {:.2f}, done {}, total {:.2f}".format(step, action, reward, done, total_rewards)
                print(infos)
                plt.title(infos)
                plt.imshow(state[0])
                plt.pause(0.001)
                plt.close()

            # Break
            if done:
                break

        return total_rewards, step

    def eval_step(self, inputs, targets, verbose=False):

        with torch.no_grad():
            score, steps = self.play(verbose=verbose)

        # Update Infos
        self.infos["ep_score"] = round(score, 2)
        self.infos["ep_steps"] = steps

        return {}, {"score": score, "steps": steps}, {}, {}

class JoinedDDPG(Model):

    def __init__(self, r_net, p_net, q_net, env, batch_size=64, gamma=0.99, tau=0.001, noise_theta=0.15, noise_std=0.2, noise_dt=0.05, buffer_size=50000, update_period=1, reward_done=None, include_done_transition=True, name="Joined Deep Deterministic Policy Gradient Model"):
        super(JoinedDDPG, self).__init__(name=name)

        # Representation Networks
        self.r_net = r_net
        self.r_target = type(self.r_net)()
        self.r_target.load_state_dict(self.r_net.state_dict())
        self.set_require_grad(self.r_target, False)

        # Policy Networks
        self.p_net = p_net
        self.p_target = type(self.p_net)()
        self.p_target.load_state_dict(self.p_net.state_dict())
        self.set_require_grad(self.p_target, False)


        # Q-Value Networks
        self.q_net = q_net
        self.q_target = type(self.q_net)()
        self.q_target.load_state_dict(self.q_net.state_dict())
        self.set_require_grad(self.q_target, False)
        self.q_frozen = type(self.q_net)()
        self.q_frozen.load_state_dict(self.q_net.state_dict())
        self.set_require_grad(self.q_frozen, False)

        # Env
        self.env = env
        self.state = self.env.reset()
        self.clip_low = self.env.env.action_space.low[0]
        self.clip_high = self.env.env.action_space.high[0]

        # Replay Buffer
        self.buffer = ReplayBuffer(
            size=buffer_size,
            state_size=self.env.state_size,
            action_size=self.env.action_size
        )

        # Noise Module
        self.noise = OrnsteinUhlenbeckProcess(num_actions=self.env.num_actions, mean=0, std=noise_std, theta=noise_theta, dt=noise_dt)

        # Training Params
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.buffer_size = buffer_size
        self.update_period = update_period
        self.reward_done = reward_done
        self.include_done_transition = include_done_transition

        # Training Infos
        self.episodes = 0
        self.running_rewards = 0.0
        self.ep_rewards = 0.0
        self.action_step = 0

    def compile(
        self, 
        optimizer="Adam",
        losses=[MeanLoss(targets_as_sign=False), MeanSquaredError()],
        loss_weights=[0.1, 1.0],
        metrics=None,
        decoders=None,
        collate_fn=CollateList(inputs_axis=[], targets_axis=[])
    ):
        # Compile Model
        super(JoinedDDPG, self).compile(
            optimizer=Adam(params=[{"params": self.r_net.parameters()}, {"params": self.p_net.parameters()}, {"params": self.q_net.parameters()}], lr=0.001) if optimizer == "Adam" else optimizer,
            losses=losses,
            loss_weights=loss_weights,
            metrics=metrics,
            decoders=decoders,
            collate_fn=collate_fn
        )

    def summary(self, show_dict=False):
        super(JoinedDDPG, self).summary()

        print("Representation Network Parameters: {:,}".format(self.num_params(self.r_net)))
        if show_dict:
            self.show_dict(self.r_net)

        print("Policy Network Parameters: {:,}".format(self.num_params(self.p_net)))
        if show_dict:
            self.show_dict(self.p_net)

        print("Q-Value Network Parameters: {:,}".format(self.num_params(self.q_net)))
        if show_dict:
            self.show_dict(self.q_net)

        print("Replay Buffer:")
        if show_dict:
            self.show_dict(self.buffer)

    def env_step(self):

        # Get State
        state = self.state.to(self.device)

        # Forward Policy Network
        with torch.no_grad():
            self.p_net.eval()
            action = self.p_net(self.r_net(state.unsqueeze(dim=0)))
            self.p_net.train()

        # Action Info
        self.infos["actions"] = ["{}{:.2f}".format("+" if a >= 0 else "-", abs(a)) for a in action.squeeze(dim=0).tolist()]

        # Update Step
        self.action_step += 1

        # Add Noise
        action += self.noise()

        # Clip Action
        action = action.clip(self.clip_low, self.clip_high)

        # Env Step
        state_next, reward, done = self.env.step(action.squeeze(dim=0))

        # Render
        #self.env.env.render()

        # Store Transitions
        if not done or self.include_done_transition:
            self.buffer.append(state, action.squeeze(dim=0), torch.tensor(reward, device=self.device, dtype=torch.float32), state_next.to(self.device), torch.tensor(done, device=self.device, dtype=torch.float32))

        # Update ep rewards
        self.ep_rewards += reward

        # Done
        if done:
            self.state = self.env.reset()
            self.episodes += 1
            self.running_rewards = 0.05 * self.ep_rewards + (1 - 0.05) * self.running_rewards
            self.ep_rewards = 0.0
            self.noise.reset()
        else:
            self.state = state_next

    def train_step(self, inputs, targets, mixed_precision, grad_scaler, accumulated_steps, acc_step):

        # Environment Step
        env_step = 0
        while env_step < self.update_period or self.buffer.num_elt < self.batch_size:
          self.env_step()
          env_step += 1

        # Sample Inputs
        inputs = self.buffer.sample(self.batch_size)

        # Update Infos
        self.infos["episodes"] = self.episodes
        self.infos["running_rewards"] = round(self.running_rewards, 2)
        self.infos["ep_rewards"] = round(self.ep_rewards, 2)
        self.infos["action_step"] = self.action_step

        # Train Step
        batch_losses, batch_metrics, _ = super(JoinedDDPG, self).train_step(inputs, targets, mixed_precision, grad_scaler, accumulated_steps, acc_step)

        # Update Target Networks
        for param_target, param_net in zip(self.r_target.parameters(), self.r_net.parameters()):
            param_target.mul_(1 - self.tau)
            param_target.add_(self.tau * param_net.detach())
        for param_target, param_net in zip(self.q_target.parameters(), self.q_net.parameters()):
            param_target.mul_(1 - self.tau)
            param_target.add_(self.tau * param_net.detach())
        for param_target, param_net in zip(self.p_target.parameters(), self.p_net.parameters()):
            param_target.mul_(1 - self.tau)
            param_target.add_(self.tau * param_net.detach())

        # Update Frozen Network
        self.q_frozen.load_state_dict(self.q_net.state_dict())

        return batch_losses, batch_metrics, _

    def forward(self, inputs):

        # Unpack Inputs
        states, actions, rewards, states_next, dones = inputs

        # Forward Representation Network
        r_states = self.r_net(states)

        # Forward Policy Network
        pred_actions = self.p_net(r_states)

        # Forward Q-Value Network
        actor_pred_returns = self.q_frozen([r_states, pred_actions]) # original
        #actor_pred_returns = self.q_target([r_states, pred_actions])

        # Forward Q-Value Network
        critic_pred_returns = self.q_net([r_states, actions])

        # Compute Q-Value Network Targets
        with torch.no_grad():
            r_states_next = self.r_target(states_next)
            if self.reward_done != None:
                self.additional_targets["critic"] = (rewards.unsqueeze(-1) + self.gamma * self.q_target([r_states_next, self.p_target(r_states_next)])) * (1 - dones.unsqueeze(-1)) + self.reward_done * dones.unsqueeze(-1)
            else:
                self.additional_targets["critic"] = rewards.unsqueeze(-1) + self.gamma * self.q_target([r_states_next, self.p_target(r_states_next)]) * (1 - dones.unsqueeze(-1)) 

        return {"actor": - actor_pred_returns, "critic": critic_pred_returns}

    def play(self, verbose=0):

        # Reset
        state = self.env.reset().to(self.device)
        total_rewards = 0
        step = 0

        # Episode loop
        while 1:

            # Sample Action
            action = self.p_net(self.r_net(state.unsqueeze(dim=0)))

            # Forward Env
            state, reward, done = self.env.step(action.squeeze(dim=0))
            state = state.to(self.device)
            step += 1
            total_rewards += reward

            # Verbose lvl 1
            if verbose > 0:
                infos = "step {}, action {}, reward {}{:.2f}, done {}, total {:.2f}".format(step, ["{}{:.2f}".format("+" if a >= 0 else "-", abs(a)) for a in action.squeeze(dim=0).tolist()], "+" if reward >=0 else "-", abs(reward), done, total_rewards)
                print(infos)

                # Image State
                if len(state.shape) == 3:
                    plt.title(infos)
                    plt.imshow(state[0].cpu())
                    plt.pause(0.001)
                    plt.close()

            # Verbose lvl 2
            if verbose > 1:
                self.env.env.render()

            # Break
            if done:
                break

        return total_rewards, step

    def eval_step(self, inputs, targets, verbose=0):

        with torch.no_grad():
            score, steps = self.play(verbose=verbose)

        # Update Infos
        self.infos["ep_score"] = round(score, 2)
        self.infos["ep_steps"] = steps

        return {}, {"score": score, "steps": steps}, {}, {}



class TouchJDDPG(Model):

    def __init__(self, r_net, p_net, q_net, env, batch_size=64, gamma=0.99, tau=0.001, noise_theta=0.15, noise_std=0.2, noise_dt=0.05, buffer_size=50000, name="Touch Joined Deep Deterministic Policy Gradient Model"):
        super(TouchJDDPG, self).__init__(name=name)

        # Representation Networks
        self.r_net = r_net
        self.r_target = type(self.r_net)()
        self.r_target.load_state_dict(self.r_net.state_dict())
        self.set_require_grad(self.r_target, False)

        # Policy Networks
        self.p_net = p_net
        self.p_target = type(self.p_net)()
        self.p_target.load_state_dict(self.p_net.state_dict())
        self.set_require_grad(self.p_target, False)


        # Q-Value Networks
        self.q_net = q_net
        self.q_target = type(self.q_net)()
        self.q_target.load_state_dict(self.q_net.state_dict())
        self.set_require_grad(self.q_target, False)
        self.q_frozen = type(self.q_net)()
        self.q_frozen.load_state_dict(self.q_net.state_dict())
        self.set_require_grad(self.q_frozen, False)


        # Env
        self.env = env
        self.state = self.env.reset()
        self.clip_low = self.env.env.action_space.low[0]
        self.clip_high = self.env.env.action_space.high[0]

        # Training Params
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.buffer_size = buffer_size
        self.noise = self.OrnsteinUhlenbeckProcess(mean=0, std_deviation=noise_std, theta=noise_theta, dt=noise_dt)

        # Training Infos
        self.episodes = 0
        self.buffer = []
        self.running_rewards = 0.0
        self.ep_rewards = 0.0

    def compile(
        self, 
        optimizer="Adam",
        losses=[MeanLoss(targets_as_sign=False), MeanSquaredError()],
        loss_weights=[0.1, 1.0],
        metrics=None,
        decoders=None,
        collate_fn=CollateList(inputs_axis=[], targets_axis=[])
    ):
        # Compile Model
        super(TouchJDDPG, self).compile(
            optimizer=Adam(params=[{"params": self.r_net.parameters()}, {"params": self.p_net.parameters()}, {"params": self.q_net.parameters()}], lr=0.001) if optimizer == "Adam" else optimizer,
            losses=losses,
            loss_weights=loss_weights,
            metrics=metrics,
            decoders=decoders,
            collate_fn=collate_fn
        )

    class OrnsteinUhlenbeckProcess:

        def __init__(self, mean, std_deviation, theta, dt=1e-2, x_initial=None):

            self.theta = theta
            self.mean = mean
            self.std_dev = std_deviation
            self.dt = dt
            self.x_initial = x_initial
            self.reset()

        def __call__(self):

            # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
            x = self.x_prev + self.theta * (self.mean - self.x_prev) * self.dt + self.std_dev * self.dt ** 0.5 * torch.randn(1)

            self.x_prev = x

            return x

        def reset(self):

            if self.x_initial is not None:
                self.x_prev = self.x_initial
            else:
                self.x_prev = 0.0

    def summary(self, show_dict=False):

        print(self.name)
        print("Model Parameters:", self.num_params(self.r_net) + self.num_params(self.p_net) + self.num_params(self.q_net))

        print("Representation Network Parameters:", self.num_params(self.r_net))
        if show_dict:
            self.show_dict(self.r_net)

        print("Policy Network Parameters:", self.num_params(self.p_net))
        if show_dict:
            self.show_dict(self.p_net)

        print("Q-Value Network Parameters:", self.num_params(self.q_net))
        if show_dict:
            self.show_dict(self.q_net)

    def env_step(self):

        # Get State
        state = [substate.to(self.device) for substate in self.state]

        # Forward Policy Network
        with torch.no_grad():
            action = self.p_net(self.r_net([substate.unsqueeze(dim=0) for substate in state]))

        # Add Noise
        action += self.noise().to(self.device)

        # Clip Action
        action = action.clip(self.clip_low, self.clip_high)

        # Env Step
        state_next, reward, done = self.env.step(action.item())

        # Limit the state and reward history
        if len(self.buffer) >= self.buffer_size:
            del self.buffer[:1]

        # Store Transitions
        self.buffer.append((state, action[0], torch.tensor(reward, device=self.device, dtype=torch.float32), [substate.to(self.device) for substate in state_next]))

        # Update ep rewards
        self.ep_rewards += reward

        # Done
        if done:
            self.state = self.env.reset()
            self.episodes += 1
            self.running_rewards = 0.05 * self.ep_rewards + (1 - 0.05) * self.running_rewards
            self.ep_rewards = 0.0
            self.noise.reset()
        else:
            self.state = state_next

    def sample_batch(self):

        # Sample Batch from Buffer
        samples = [self.buffer[i] for i in torch.randint(low=0, high=len(self.buffer), size=(self.batch_size,))]
        states = [torch.stack([sample[0][substate_id] for sample in samples]) for substate_id in range(len(self.state))]
        actions = torch.stack([sample[1] for sample in samples])
        rewards = torch.stack([sample[2] for sample in samples])
        states_next = [torch.stack([sample[3][substate_id] for sample in samples]) for substate_id in range(len(self.state))]

        return [states, actions, rewards, states_next]

    def train_step(self, inputs, targets, mixed_precision, grad_scaler, accumulated_steps, acc_step):

        # Environment Step
        self.env_step()

        # Sample Inputs / Targets Batch
        inputs = self.sample_batch()

        # Update Infos
        self.infos["episodes"] = self.episodes
        self.infos["running_rewards"] = round(self.running_rewards, 2)
        self.infos["ep_rewards"] = round(self.ep_rewards, 2)

        # Train Step
        batch_losses, batch_metrics, _ = super(TouchJDDPG, self).train_step(inputs, targets, mixed_precision, grad_scaler, accumulated_steps, acc_step)

        # Update Target Networks
        for param_target, param_net in zip(self.r_target.parameters(), self.r_net.parameters()):
            param_target.mul_(1 - self.tau)
            param_target.add_(self.tau * param_net.detach())
        for param_target, param_net in zip(self.q_target.parameters(), self.q_net.parameters()):
            param_target.mul_(1 - self.tau)
            param_target.add_(self.tau * param_net.detach())
        for param_target, param_net in zip(self.p_target.parameters(), self.p_net.parameters()):
            param_target.mul_(1 - self.tau)
            param_target.add_(self.tau * param_net.detach())

        # Update Frozen Network
        self.q_frozen.load_state_dict(self.q_net.state_dict())

        return batch_losses, batch_metrics, _

    def forward(self, inputs):

        # Unpack Inputs
        states, actions, rewards, states_next = inputs

        # Forward Representation Network
        r_states = self.r_net(states)

        # Forward Policy Network
        pred_actions = self.p_net(r_states)

        # Forward Q-Value Network
        actor_pred_returns = self.q_frozen([r_states, pred_actions])

        # Forward Q-Value Network
        critic_pred_returns = self.q_net([r_states, actions])

        # Compute Q-Value Network Targets
        with torch.no_grad():
            r_states_next = self.r_target(states_next)
            self.additional_targets["critic"] = rewards.unsqueeze(-1) + self.gamma * self.q_target([r_states_next, self.p_target(r_states_next)])

        return {"actor": - actor_pred_returns, "critic": critic_pred_returns}

    def play(self, verbose=False):

        # Reset
        state = [substate.to(self.device) for substate in self.env.reset()]
        total_rewards = 0
        step = 0

        # Episode loop
        while 1:

            # Sample Action
            action = self.p_net(self.r_net([substate.unsqueeze(dim=0) for substate in state]))

            # Forward Env
            state, reward, done = self.env.step(action)
            step += 1
            total_rewards += reward

            # Verbose
            if verbose:
                infos = "step {}, action {}, reward {:.2f}, done {}, total {:.2f}".format(step, action, reward, done, total_rewards)
                print(infos)

            # Break
            if done:
                break

        return total_rewards, step

    def eval_step(self, inputs, targets, verbose=False):

        with torch.no_grad():
            score, steps = self.play(verbose=verbose)

        # Update Infos
        self.infos["ep_score"] = round(score, 2)
        self.infos["ep_steps"] = steps

        return {}, {"score": score, "steps": steps}, {}, {}
































































































class JoinedDDPG0(Model):

    def __init__(self, r_net, p_net, q_net, env, batch_size=64, gamma=0.99, tau=0.001, noise_theta=0.15, noise_std=0.2, noise_dt=0.05, buffer_size=50000, name="Joined Deep Deterministic Policy Gradient Model"):
        super(JoinedDDPG0, self).__init__(name=name)

        # Representation Networks
        self.r_net = r_net
        self.r_target = type(self.r_net)()
        self.r_target.load_state_dict(self.r_net.state_dict())
        self.set_require_grad(self.r_target, False)

        # Policy Networks
        self.p_net = p_net
        self.p_target = type(self.p_net)()
        self.p_target.load_state_dict(self.p_net.state_dict())
        self.set_require_grad(self.p_target, False)


        # Q-Value Networks
        self.q_net = q_net
        self.q_target = type(self.q_net)()
        self.q_target.load_state_dict(self.q_net.state_dict())
        self.set_require_grad(self.q_target, False)

        # Critic
        self.critic = self.Critic(r_net=self.r_net, q_net=self.q_net, r_target=self.r_target, p_target=self.p_target, q_target=self.q_target, gamma=gamma)

        # Actor
        self.actor = self.Actor(r_net=self.r_net, p_net=self.p_net, q_net=self.q_net)

        # Env
        self.env = env
        self.state = self.env.reset()
        self.clip_low = self.env.env.action_space.low[0]
        self.clip_high = self.env.env.action_space.high[0]

        # Training Params
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.buffer_size = buffer_size
        self.noise = self.OrnsteinUhlenbeckProcess(mean=0, std_deviation=noise_std, theta=noise_theta, dt=noise_dt)

        # Training Infos
        self.episodes = 0
        self.buffer = []
        self.running_rewards = 0.0
        self.ep_rewards = 0.0

    def compile(
        self, 

        critic_optimizer="Adam",
        critic_losses=MeanSquaredError(),
        critic_loss_weights=None,
        critic_metrics=None,
        critic_decoders=None,
        critic_collate_fn=None,

        actor_optimizer="Adam",
        actor_losses=MeanLoss(targets_as_sign=False),
        actor_loss_weights=None,
        actor_metrics=None,
        actor_decoders=None,
        actor_collate_fn=None
    ):
        # Compile Critic
        self.critic.compile(
            optimizer=Adam(params=[{"params": self.r_net.parameters()}, {"params": self.q_net.parameters()}], lr=0.001) if critic_optimizer == "Adam" else critic_optimizer,
            losses=critic_losses,
            loss_weights=critic_loss_weights,
            metrics=critic_metrics,
            decoders=critic_decoders,
            collate_fn=critic_collate_fn
        )

        # Compile Actor
        self.actor.compile(
            optimizer=Adam(params=[{"params": self.r_net.parameters()}, {"params": self.p_net.parameters()}], lr=0.0001) if actor_optimizer == "Adam" else actor_optimizer,
            losses=actor_losses,
            loss_weights=actor_loss_weights,
            metrics=actor_metrics,
            decoders=actor_decoders,
            collate_fn=actor_collate_fn
        )

        # Model Step
        self.model_step = self.actor.optimizer.scheduler.model_step

        # Collate Function
        self.collate_fn = CollateList(inputs_axis=[], targets_axis=[])

        # Optimizer
        self.optimizer = {"critic": self.critic.optimizer, "actor": self.actor.optimizer}

        # Set Compiled to True
        self.compiled = True

    class OrnsteinUhlenbeckProcess:

        def __init__(self, mean, std_deviation, theta, dt=1e-2, x_initial=None):

            self.theta = theta
            self.mean = mean
            self.std_dev = std_deviation
            self.dt = dt
            self.x_initial = x_initial
            self.reset()

        def __call__(self):

            # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
            x = self.x_prev + self.theta * (self.mean - self.x_prev) * self.dt + self.std_dev * self.dt ** 0.5 * torch.randn(1)

            self.x_prev = x

            return x

        def reset(self):

            if self.x_initial is not None:
                self.x_prev = self.x_initial
            else:
                self.x_prev = 0.0

    def summary(self, show_dict=False):

        print(self.name)
        print("Model Parameters:", self.num_params(self.r_net) + self.num_params(self.p_net) + self.num_params(self.q_net))

        print("Representation Network Parameters:", self.num_params(self.r_net))
        if show_dict:
            self.show_dict(self.r_net)

        print("Policy Network Parameters:", self.num_params(self.p_net))
        if show_dict:
            self.show_dict(self.p_net)

        print("Q-Value Network Parameters:", self.num_params(self.q_net))
        if show_dict:
            self.show_dict(self.q_net)

    def env_step(self):

        # Get State
        state = self.state.to(self.device)

        # Forward Policy Network
        with torch.no_grad():
            action = self.p_net(self.r_net(state.unsqueeze(dim=0)))

        # Add Noise
        action += self.noise().to(self.device)

        # Clip Action
        action = action.clip(self.clip_low, self.clip_high)

        # Env Step
        state_next, reward, done = self.env.step(action.item())

        # Limit the state and reward history
        if len(self.buffer) >= self.buffer_size:
            del self.buffer[:1]

        # Store Transitions
        self.buffer.append((state, action[0], torch.tensor(reward, device=self.device, dtype=torch.float32), state_next.to(self.device)))

        # Update ep rewards
        self.ep_rewards += reward

        # Done
        if done:
            self.state = self.env.reset()
            self.episodes += 1
            self.running_rewards = 0.05 * self.ep_rewards + (1 - 0.05) * self.running_rewards
            self.ep_rewards = 0.0
            self.noise.reset()
        else:
            self.state = state_next

    def sample_batch(self):

        # Sample Batch from Buffer
        samples = [self.buffer[i] for i in torch.randint(low=0, high=len(self.buffer), size=(self.batch_size,))]
        states = torch.stack([sample[0] for sample in samples])
        actions = torch.stack([sample[1] for sample in samples])
        rewards = torch.stack([sample[2] for sample in samples])
        states_next = torch.stack([sample[3] for sample in samples])

        # Create Batches
        critic_inputs = [states, actions, rewards, states_next]
        actor_inputs = states

        return critic_inputs, actor_inputs

    def train_step(self, inputs, targets, mixed_precision, grad_scaler, accumulated_steps, acc_step):

        # Environment Step
        self.env_step()

        # Sample Inputs / Targets Batch
        critic_inputs, actor_inputs = self.sample_batch()

        # Init Dict
        batch_losses = {}
        batch_metrics = {}

        # Critic Step
        self.set_require_grad(self.q_net, True)
        critic_batch_losses, critic_batch_metrics, _ = self.critic.train_step(critic_inputs, targets, mixed_precision, grad_scaler, accumulated_steps, acc_step)
        batch_losses.update({"critic_" + key: value for key, value in critic_batch_losses.items()})
        batch_metrics.update({"critic_" + key: value for key, value in critic_batch_metrics.items()})

        # Actor Step
        self.set_require_grad(self.q_net, False)
        actor_batch_losses, actor_batch_metrics, _ = self.actor.train_step(actor_inputs, targets, mixed_precision, grad_scaler, accumulated_steps, acc_step)
        batch_losses.update({"actor_" + key: value for key, value in actor_batch_losses.items()})
        batch_metrics.update({"actor_" + key: value for key, value in actor_batch_metrics.items()})

        # Update Target Networks
        for param_target, param_net in zip(self.r_target.parameters(), self.r_net.parameters()):
            param_target.mul_(1 - self.tau)
            param_target.add_(self.tau * param_net.detach())
        for param_target, param_net in zip(self.q_target.parameters(), self.q_net.parameters()):
            param_target.mul_(1 - self.tau)
            param_target.add_(self.tau * param_net.detach())
        for param_target, param_net in zip(self.p_target.parameters(), self.p_net.parameters()):
            param_target.mul_(1 - self.tau)
            param_target.add_(self.tau * param_net.detach())

        # Update Infos
        self.infos["episodes"] = self.episodes
        self.infos["running_rewards"] = round(self.running_rewards, 2)
        self.infos["ep_rewards"] = round(self.ep_rewards, 2)
        self.infos["step"] = self.model_step
        self.infos["critic_lr"] = self.critic.optimizer.param_groups[0]["lr"]
        self.infos["actor_lr"] = self.actor.optimizer.param_groups[0]["lr"]

        return batch_losses, batch_metrics, _

    class Critic(Model):

        def __init__(self, r_net, q_net, r_target, p_target, q_target, gamma):
            super().__init__(name="Critic Model")

            self.r_net = r_net
            self.q_net = q_net
            self.r_target = r_target
            self.p_target = p_target
            self.q_target = q_target
            self.gamma = gamma

        def forward(self, inputs):
            
            # Unpack Inputs
            states, actions, rewards, states_next = inputs

            # Forward Q-Value network
            pred_returns = self.q_net([self.r_net(states), actions])

            # Compute Targets
            with torch.no_grad():
                r_states = self.r_target(states_next)
                self.additional_targets["value"] = rewards.unsqueeze(-1) + self.gamma * self.q_target([r_states, self.p_target(r_states)])

            return {"value": pred_returns}
    
    class Actor(Model):

        def __init__(self, r_net, p_net, q_net):
            super().__init__(name="Actor Model")

            self.r_net = r_net
            self.p_net = p_net
            self.q_net = q_net

        def forward(self, inputs):
            
            # Unpack Inputs
            states = inputs

            # Forward Representation Network
            r_states = self.r_net(states)

            # Forward Policy Network
            pred_actions = self.p_net(r_states)

            # Forward Q-Value network
            pred_returns = self.q_net([r_states, pred_actions])

            return - pred_returns

    def play(self, verbose=False):

        # Reset
        state = self.env.reset().to(self.device)
        total_rewards = 0
        step = 0

        # Episode loop
        while 1:

            # Sample Action
            action = self.p_net(self.r_net(state.unsqueeze(dim=0)))

            # Forward Env
            state, reward, done = self.env.step(action)
            step += 1
            total_rewards += reward

            # Verbose
            if verbose:
                infos = "step {}, action {}, reward {}, done {}, total {}".format(step, action, reward, done, total_rewards)
                print(infos)
                plt.title(infos)
                plt.imshow(state[0])
                plt.pause(0.001)
                plt.close()

            # Break
            if done:
                break

        return total_rewards, step

    def eval_step(self, inputs, targets, verbose=False):

        with torch.no_grad():
            score, steps = self.play(verbose=verbose)

        # Update Infos
        self.infos["ep_score"] = score
        self.infos["ep_steps"] = steps

        return {}, {"score": score, "steps": steps}, {}, {}


class OnlineActorCritic(Model):

    def __init__(self, p_net, p_target, q_net, q_target, env, max_steps=10000, gamma=0.99, eps_min=0.1, eps_max=1.0, eps_random=50000, eps_steps=1000000, max_memory_length=100000, update_target_network_period=10000, name="Online Actor Critic Network"):
        super(OnlineActorCritic, self).__init__(name=name)

        # Policy Networks
        self.p_net = p_net
        self.p_target = p_target

        # Q-Value Networks
        self.q_net = q_net
        self.q_target = q_target

        # Actor
        self.actor = self.Actor(p_net=p_net, q_target=q_target)

        # Critic
        self.critic = self.Critic(p_target=p_target, q_net=q_net, q_target=q_target)

        # Learning Params
        self.max_steps = max_steps
        self.gamma = gamma
        self.eps_min = eps_min
        self.eps = eps_max
        self.eps_inter = eps_max - eps_min
        self.eps_random = eps_random
        self.eps_steps = eps_steps
        self.update_target_network_period = update_target_network_period
        self.max_memory_length = max_memory_length

        # Training Params
        self.updated = False
        self.done = False
        self.action_step = 0
        self.episodes = 0
        self.buffer = []
        self.running_rewards = 0.0

    def compile(
        self, 

        actor_optimizer="Adam",
        actor_losses=MeanLoss(),
        actor_loss_weights=None,
        actor_metrics=None,
        actor_decoders=None,
        actor_collate_fn=None,

        critic_optimizer="Adam",
        critic_losses=HuberLoss(),
        critic_loss_weights=None,
        critic_metrics=None,
        critic_decoders=None,
        critic_collate_fn=None,
    ):

        # Compile Actor
        self.actor.compile(
            optimizer=Adam(params=self.p_net.parameters(), lr=0.00025) if actor_optimizer == "Adam" else actor_optimizer,
            losses=actor_losses,
            loss_weights=actor_loss_weights,
            metrics=actor_metrics,
            decoders=actor_decoders,
            collate_fn=None
        )

        # Compile Critic
        self.critic.compile(
            optimizer=Adam(params=self.q_net.parameters(), lr=0.00025) if critic_optimizer == "Adam" else critic_optimizer,
            losses=critic_losses,
            loss_weights=critic_loss_weights,
            metrics=critic_metrics,
            decoders=critic_decoders,
            collate_fn=None
        )

        # Collate Function
        self.collate_fn = CollateList(inputs_axis=[], targets_axis=[])

        # Optimizer: To Review
        self.optimizer = {"actor": self.actor.optimizer, "critic": self.critic.optimizer}

        # Set Compiled to True
        self.compiled = True

    def summary(self, show_dict=False):

        print(self.name)
        print("Model Parameters:", self.num_params(self.actor) + self.num_params(self.critic))

        print("Actor Parameters:", self.num_params(self.actor))
        if show_dict:
            self.show_dict(self.actor)

        print("Critic Parameters:", self.num_params(self.critic))
        if show_dict:
            self.show_dict(self.critic)

    def train_step(self, inputs, targets, mixed_precision, grad_scaler, accumulated_steps, acc_step):

        # Init Dict
        batch_losses = {}
        batch_metrics = {}

        # Critic Step
        self.set_require_grad(self.critic, True)
        critic_batch_losses, critic_batch_metrics, _ = self.critic.train_step(inputs, targets, mixed_precision, grad_scaler, accumulated_steps, acc_step)
        batch_losses.update({"critic_" + key: value for key, value in critic_batch_losses.items()})
        batch_metrics.update({"critic_" + key: value for key, value in critic_batch_metrics.items()})

        # Actor Step
        self.set_require_grad(self.critic, False)
        actor_batch_losses, actor_batch_metrics, _ = self.actor.train_step(inputs, targets, mixed_precision, grad_scaler, accumulated_steps, acc_step)
        batch_losses.update({"actor_" + key: value for key, value in actor_batch_losses.items()})
        batch_metrics.update({"actor_" + key: value for key, value in actor_batch_metrics.items()})

        return batch_losses, batch_metrics, _

    class Critic(Model):

        def __init__(self, p_net, q_net):
            super().__init__()
            self.p_net = p_net
            self.q_net = q_net

        def forward(self, inputs):
            pass


class OnlineActorCritic(Model):

    def __init__(self, p_net, p_target, q_net, q_target, env, max_steps=10000, gamma=0.99, eps_min=0.1, eps_max=1.0, eps_random=50000, eps_steps=1000000, update_target_network_period=10000, name="Online Actor Critic Network"):
        super(OnlineActorCritic, self).__init__(name=name)

        # Policy Networks
        self.p_net = p_net
        self.set_require_grad(self.p_net, False)
        self.p_target = p_target
        self.set_require_grad(self.p_target, False)

        # Q Value Networks
        self.q_net = q_net
        self.q_target = q_target
        self.set_require_grad(self.q_target, False)

        # Learning Params
        self.max_steps = max_steps
        self.gamma = gamma
        self.eps_min = eps_min
        self.eps = eps_max
        self.eps_inter = eps_max - eps_min
        self.eps_random = eps_random
        self.eps_steps = eps_steps
        self.update_target_network_period = update_target_network_period

        # Infos
        self.episodes = 0
        self.ep_rewards = 0.0
        self.running_rewards = 0.0

        # Init State
        self.env = env
        self.state = self.env.reset()

    def compile(
        self, 
        optimizer="Adam",
        losses={"actor": MeanLoss(targets_as_sign=False), "critic": HuberLoss()},
        loss_weights=[1, 1],
        metrics=None,
        decoders=None,
        collate_fn=CollateList(inputs_axis=[], targets_axis=[])
    ):

        super(OnlineActorCritic, self).compile(
            optimizer=Adam(
                params=[{"params": self.p_net.parameters()}, {"params": self.q_net.parameters()}], 
                lr=0.00025
            ) if optimizer == "Adam" else optimizer,
            losses=losses,
            loss_weights=loss_weights,
            metrics=metrics,
            decoders=decoders,
            collate_fn=collate_fn
        )

    def summary(self, show_dict=False):

        print(self.name)
        print("Model Parameters:", self.num_params(self.p_net) + self.num_params(self.q_net))

        print("Actor Parameters:", self.num_params(self.p_net))
        if show_dict:
            self.show_dict(self.p_net)

        print("Critic Parameters:", self.num_params(self.q_net))
        if show_dict:
            self.show_dict(self.q_net)

    def train_step(self, inputs, targets, mixed_precision, grad_scaler, accumulated_steps, acc_step):

        # Train Step
        batch_losses, batch_metrics, _ = super(OnlineActorCritic, self).train_step(inputs, targets, mixed_precision, grad_scaler, accumulated_steps, acc_step)

        # Update Targets Networks
        if self.model_step % self.update_target_network_period == 0:
            #print("update")
            self.p_target.load_state_dict(self.p_net.state_dict())
            self.q_target.load_state_dict(self.q_net.state_dict())
            self.set_require_grad(self.p_net, True)

        # Decay epsilon
        self.eps -= self.eps_inter / self.eps_steps
        self.eps = max(self.eps, self.eps_min)

        #print(self.p_net.linear2.weight)
        #print(self.p_target.linear2.weight)
        #print(self.q_net.linear2.weight)
        #print(self.q_target.linear2.weight)

        return batch_losses, batch_metrics, _

    def forward(self, inputs):

        # Unpack Inputs
        #state = inputs
        state = self.state.to(self.device)

        # Forward Policy Network
        action_pred = self.p_net(state.unsqueeze(dim=0))
        self.infos["action_pred"] = [round(elt, 2) for elt in action_pred.detach().cpu().tolist()[0]]

        # Forward Q value Target Network
        p_loss = - self.q_target(state.unsqueeze(dim=0), action_pred)

        # Sample Action
        with torch.no_grad():
            if self.model_step < self.eps_random or self.eps > torch.rand(1).item():
                action_prob = torch.rand(1, self.env.actions, device=self.device).softmax(dim=-1)
            else:
                action_prob = self.p_target(state.unsqueeze(dim=0))

            action = action_prob.multinomial(num_samples=1).item()

        # Env Step
        state_next, reward, done = self.env.step(action)

        # Done
        if done:
            self.episodes += 1
            self.running_rewards = 0.05 * self.ep_rewards + (1 - 0.05) * self.running_rewards
            self.ep_rewards = 0.0
            self.state = self.env.reset()
        else:
            self.ep_rewards += reward
            self.state = state_next
            state_next = state_next.to(self.device)

        # Forward Q Value Network
        returns_pred = self.q_net(state.unsqueeze(dim=0), action_prob)
        self.infos["returns_pred"] = round(returns_pred.detach().item(), 2)

        # Compute Q Value Target
        if not done:
            with torch.no_grad():

                # Sample Next Action
                if self.model_step < self.eps_random or self.eps > torch.rand(1).item():
                    action_prob_next = torch.rand(1, self.env.actions, device=self.device)#.softmax(dim=-1)
                else:
                    action_prob_next = self.p_target(state_next.unsqueeze(dim=0))

                # Forward Q Value Target Network
                returns_next = self.q_target(state_next.unsqueeze(dim=0), action_prob_next)

                # Compute Q Value Target
                returns_target = reward + self.gamma * returns_next
        else:
            returns_target = - torch.ones(1, 1, device=self.device)

        # Add targets
        self.additional_targets["critic"] = returns_target

        # Update Infos
        self.infos["episodes"] = self.episodes
        self.infos["running_rewards"] = round(self.running_rewards, 2)
        self.infos["eps"] = round(self.eps, 2)
        self.infos["ep_rewards"] = self.ep_rewards

        return {"actor": p_loss, "critic": returns_pred}

class SeqToSeqModel(Model):

    def __init__(self, params):
        super(SeqToSeqModel, self).__init__(params)

        # Encoder Networks
        self.encoder = nn.ModuleList()
        for network, network_params  in params["encoder_params"].items():
            self.encoder.append(networks_dict[network](network_params))

        # Decoder Networks
        self.decoder = nn.ModuleList()
        for network, network_params  in params["decoder_params"].items():
            self.decoder.append(networks_dict[network](network_params))

        # ids
        self.padding_id = params["training_params"]["padding_id"]
        self.start_id = params["training_params"]["start_id"]
        self.stop_id = params["training_params"]["stop_id"]

        # Criterion
        self.criterion = SoftmaxCrossEntropyLoss(ignore_index=self.padding_id, transpose_logits=True)

        # Metric
        self.metric = CategoricalAccuracy(ignore_index=self.padding_id)

        # Compile
        self.compile(params["training_params"])

    def forward(self, inputs):

        # Encoder Networks
        for network in self.encoder:
            inputs.update(network(inputs))

        # Decoder Networks
        for network in self.decoder:
            inputs.update(network(inputs))

        return inputs

    def decode(self, outputs, from_logits=True):

        if from_logits:
            # Softmax -> argmax
            tokens = outputs["samples"].softmax(dim=-1).argmax(axis=-1).tolist()
        else:
            tokens = outputs["samples"].tolist()

        return tokens

    def collate(self, samples):

        # Init
        inputs = {}
        targets = {}

        # Inputs Encoder
        inputs_samples = [sample[0] for sample in samples]
        inputs["lengths"] = torch.tensor([len(sample) for sample in inputs_samples], dtype=torch.long)
        inputs["samples"] = torch.nn.utils.rnn.pad_sequence(inputs_samples, batch_first=True, padding_value=self.padding_id)
        
        # Inputs Decoder
        inputs_samples_dec = [torch.nn.functional.pad(sample[1], pad=(1, 0), value=self.start_id) for sample in samples]
        inputs["lengths_dec"] = torch.tensor([len(sample) for sample in inputs_samples_dec], dtype=torch.long)
        inputs["samples_dec"] = torch.nn.utils.rnn.pad_sequence(inputs_samples_dec, batch_first=True, padding_value=self.padding_id)

        # Targets
        targets_samples = [torch.nn.functional.pad(sample[1], pad=(0, 1), value=self.stop_id) for sample in samples]
        targets["samples"] = torch.nn.utils.rnn.pad_sequence(targets_samples, batch_first=True, padding_value=self.padding_id)

        #print(inputs, targets)
        #exit()

        return {"inputs": inputs, "targets": targets}

    def summary(self, show_dict=False):

        print(self.name)
        print("Model Parameters:", self.num_params() - self.lm.num_params() if self.lm else self.num_params())
        print(" - Encoder Parameters:", sum([p.numel() for p in self.encoder.parameters()]))
        print(" - Decoder Parameters:", sum([p.numel() for p in self.decoder.parameters()]))

        if self.lm:
            print("LM Parameters:", self.lm.num_params())

        if show_dict:
            state_dict = self.state_dict()
            max_len = max([len(key) for key in state_dict.keys()]) + 5
            for key, value in state_dict.items():
                print("param: {} shape: {:<16} mean: {:<12.4f} std: {:<12.4f} dtype: {:<12}".format(key + " " * (max_len - len(key)), str(tuple(value.size())), value.float().mean(), value.float().std(), str(value.dtype)))

class Transducer(Model):

    def __init__(self, params):
        super(Transducer, self).__init__(params)

        # Encoder Networks
        self.encoder = nn.ModuleList()
        for network, network_params  in params["encoder_params"].items():
            self.encoder.append(networks_dict[network](network_params))

        # Decoder Networks
        self.decoder = nn.ModuleList()
        for network, network_params  in params["decoder_params"].items():
            self.decoder.append(networks_dict[network](network_params))

        # Joint Network
        self.joint_network = JointNetwork(params["joint_params"])

        # Init VN
        #self.decoder.apply(lambda m: init_vn(m, params["training_params"].get("vn_std", None)))

        # ids
        self.blank_id = params["training_params"]["blank_id"]
        self.padding_id = params["training_params"]["padding_id"]
        self.start_id = params["training_params"]["start_id"]

        # Criterion
        self.criterion = RNNTLoss(blank=self.blank_id)

        # Metric
        self.metric = None

        # Decoding
        self.max_consec_dec_step = params["decoder_params"].get("max_consec_dec_step", 5)

        # Compile
        self.compile(params["training_params"])

    def forward(self, inputs):

        # Encoder Network
        for network in self.encoder:
            inputs.update(network(inputs))

        # Update Inputs
        inputs["samples_enc"], inputs["lengths_enc"], inputs["hidden_enc"] = inputs["samples"], inputs["lengths"], inputs["hidden"]
        inputs["samples"], inputs["lengths"] = inputs["samples_dec"], inputs["lengths_dec"]
        inputs["hidden"] = None

        # Decoder Network
        for network in self.decoder:
            inputs.update(network(inputs))

        # Update Inputs
        inputs["samples_dec"] = inputs["samples"]

        # Joint Network
        inputs.update(self.joint_network(inputs))

        return inputs

    def collate(self, samples):

        # Init
        inputs = {}
        targets = {}

        # Inputs Encoder
        inputs_samples = [sample[0] for sample in samples]
        inputs["lengths"] = torch.tensor([len(sample) for sample in inputs_samples], dtype=torch.long)
        inputs["samples"] = torch.nn.utils.rnn.pad_sequence(inputs_samples, batch_first=True, padding_value=self.padding_id)
        
        # Inputs Decoder
        inputs_samples_dec = [torch.nn.functional.pad(sample[1], pad=(1, 0), value=self.start_id) for sample in samples]
        inputs["lengths_dec"] = torch.tensor([len(sample) for sample in inputs_samples_dec], dtype=torch.long)
        inputs["samples_dec"] = torch.nn.utils.rnn.pad_sequence(inputs_samples_dec, batch_first=True, padding_value=self.padding_id)

        # Targets
        targets_samples = [sample[1] for sample in samples]
        targets["lengths"] = torch.tensor([len(sample) for sample in targets_samples], dtype=torch.long)
        targets["samples"] = torch.nn.utils.rnn.pad_sequence(targets_samples, batch_first=True, padding_value=self.padding_id)

        #print(inputs)
        #print(targets)
        #exit()

        return {"inputs": inputs, "targets": targets}

    def summary(self, show_dict=False):

        print(self.name)
        print("Model Parameters:", self.num_params() - self.lm.num_params() if isinstance(self.lm, LanguageModel) else self.num_params())
        print(" - Encoder Parameters:", sum([p.numel() for p in self.encoder.parameters()]))
        print(" - Decoder Parameters:", sum([p.numel() for p in self.decoder.parameters()]))
        print(" - Joint Parameters:", sum([p.numel() for p in self.joint_network.parameters()]))

        if isinstance(self.lm, LanguageModel):
            print("LM Parameters:", self.lm.num_params())

        if show_dict:
            state_dict = self.state_dict()
            max_len = max([len(key) for key in state_dict.keys()]) + 5
            for key, value in state_dict.items():
                print("param: {} shape: {:<16} mean: {:<12.4f} std: {:<12.4f} dtype: {:<12}".format(key + " " * (max_len - len(key)), str(tuple(value.size())), value.float().mean(), value.float().std(), str(value.dtype)))

    def gready_search_decoding(self, x, x_len):

        # Predictions String List
        preds = []

        # Forward Encoder (B, Taud) -> (B, T, Denc)
        f, f_len, _ = self.encoder(x, x_len)

        # Batch loop
        for b in range(x.size(0)): # One sample at a time for now, not batch optimized

            # Init y and hidden state
            y = x.new_zeros(1, 1, dtype=torch.long)
            hidden = None

            enc_step = 0
            consec_dec_step = 0

            # Decoder loop
            while enc_step < f_len[b]:

                # Forward Decoder (1, 1) -> (1, 1, Ddec)
                g, hidden = self.decoder(y[:, -1:], hidden)
                
                # Joint Network loop
                while enc_step < f_len[b]:

                    # Forward Joint Network (1, 1, Denc) and (1, 1, Ddec) -> (1, V)
                    logits = self.joint_network(f[b:b+1, enc_step], g[:, 0])

                    # Token Prediction
                    pred = logits.softmax(dim=-1).log().argmax(dim=-1) # (1)

                    # Null token or max_consec_dec_step
                    if pred == 0 or consec_dec_step == self.max_consec_dec_step:
                        consec_dec_step = 0
                        enc_step += 1
                    # Token
                    else:
                        consec_dec_step += 1
                        y = torch.cat([y, pred.unsqueeze(0)], axis=-1)
                        break

            # Decode Label Sequence
            pred = self.tokenizer.decode(y[:, 1:].tolist())
            preds += pred

        return preds

    def beam_search_decoding(self, x, x_len, beam_size=None):

        # Overwrite beam size
        if beam_size is None:
            beam_size = self.beam_size

        # Load ngram lm
        ngram_lm = None
        if self.ngram_path is not None:
            try:
                ngram_lm = kenlm.Model(self.ngram_path)
            except:
                print("Ngram language model not found...")

        # Predictions String List
        batch_predictions = []

        # Forward Encoder (B, Taud) -> (B, T, Denc)
        f, f_len, _ = self.encoder(x, x_len)

        # Batch loop
        for b in range(x.size(0)):

            # Decoder Input
            y = torch.ones((1, 1), device=x.device, dtype=torch.long)

            # Default Beam hypothesis
            B_hyps = [{
                "prediction": [0],
                "logp_score": 0.0,
                "hidden_state": None,
                "hidden_state_lm": None,
            }]

            # Init Ngram LM State
            if ngram_lm and self.ngram_alpha > 0:
                state1 = kenlm.State()
                state2 = kenlm.State()
                ngram_lm.NullContextWrite(state1)
                B_hyps[0].update({"ngram_lm_state1": state1, "ngram_lm_state2": state2})

            # Encoder loop
            for enc_step in range(f_len[b]):

                A_hyps = B_hyps
                B_hyps = []
                
                # While B contains less than W hypothesis
                while len(B_hyps) < beam_size:

                    # A most probable hyp
                    A_best_hyp = max(A_hyps, key=lambda x: x["logp_score"] / len(x["prediction"]))

                    # Remove best hyp from A
                    A_hyps.remove(A_best_hyp)

                    # Forward Decoder (1, 1) -> (1, 1, Ddec)
                    y[0, 0] = A_best_hyp["prediction"][-1]
                    g, hidden = self.decoder(y, A_best_hyp["hidden_state"])
                    g = g[:, 0] # (1, Ddec)

                    # Forward Joint Network (1, Denc) and (1, Ddec) -> (1, V)
                    logits = self.joint_network(f[b:b+1, enc_step], g)
                    logits = logits[0] # (V)

                    # Apply Temperature
                    logits = logits / self.tmp

                    # Compute logP
                    logP = logits.softmax(dim=-1).log()

                    # LM Prediction
                    if self.lm and self.lm_weight:

                        # Forward LM
                        logits_lm, hidden_lm = self.lm.decode(y, A_best_hyp["hidden_state_lm"]) # (1, 1, V)
                        logits_lm = logits_lm[0, 0] # (V)

                        # Apply Temperature
                        logits_lm = logits_lm / self.lm_tmp

                        # Compute logP
                        logP_lm = logits_lm.softmax(dim=-1).log()

                        # Add LogP
                        logP += self.lm_weight * logP_lm

                    # Sorted top k logp and their labels
                    topk_logP, topk_labels = torch.topk(logP, k=beam_size, dim=-1)

                    # Extend hyp by selection
                    for j in range(topk_logP.size(0)):

                        # Updated hyp with logp
                        hyp = {
                            "prediction": A_best_hyp["prediction"][:],
                            "logp_score": A_best_hyp["logp_score"] + topk_logP[j],
                            "hidden_state": A_best_hyp["hidden_state"],
                        }

                        # Blank Prediction -> Append hyp to B
                        if topk_labels[j] == 0:

                            if self.lm and self.lm_weight > 0:
                                hyp["hidden_state_lm"] = A_best_hyp["hidden_state_lm"]

                            if ngram_lm and self.ngram_alpha > 0:
                                hyp["ngram_lm_state1"] = A_best_hyp["ngram_lm_state1"].__deepcopy__()
                                hyp["ngram_lm_state2"] = A_best_hyp["ngram_lm_state2"].__deepcopy__()

                            B_hyps.append(hyp)

                        # Non Blank Prediction -> Update hyp hidden / prediction and append to A
                        else:
                            hyp["prediction"].append(topk_labels[j].item())
                            hyp["hidden_state"] = hidden

                            if self.lm and self.lm_weight > 0:
                                hyp["hidden_state_lm"] = hidden_lm

                            # Ngram LM Rescoring
                            if ngram_lm and self.ngram_alpha > 0:
                                
                                state1 = A_best_hyp["ngram_lm_state1"].__deepcopy__()
                                state2 = A_best_hyp["ngram_lm_state2"].__deepcopy__()
                                s = chr(topk_labels[j].item() + self.ngram_offset)
                                lm_score = ngram_lm.BaseScore(state1, s, state2)
                                hyp["logp_score"] += self.ngram_alpha * lm_score + self.ngram_beta
                                hyp["ngram_lm_state1"] = state2
                                hyp["ngram_lm_state2"] = state1
                            
                            A_hyps.append(hyp)               

            # Pick best hyp
            best_hyp = max(B_hyps, key=lambda x: x["logp_score"] / len(x["prediction"]))

            # Decode hyp
            batch_predictions.append(self.tokenizer.decode(best_hyp["prediction"][1:]))

        return batch_predictions

###############################################################################
# Model Dictionary
###############################################################################

model_dict = {
    "Classifier": Classifier,
    "GAN": GenerativeAdversarialNetwork,
    "S2S": SeqToSeqModel,
    "Transducer": Transducer
}