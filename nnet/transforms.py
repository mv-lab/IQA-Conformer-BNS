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
import torchvision

###############################################################################
# Transforms
###############################################################################

class RandomResizeShorterSide(nn.Module):

    """ RandomResizeShorterSide

    Infos: The image is resized with its shorter side ran- domly sampled in [min_size, max_size] for scale augmentation.

    Reference: "Deep Residual Learning for Image Recognition", He et al.
    https://arxiv.org/abs/1512.03385
    
    """

    def __init__(self, min_size=256, max_size=480):
        super().__init__()

        self.min_size = min_size
        self.max_size = max_size

    def forward(self, image):

        # Sample Size
        size = torch.randint(low=self.min_size, high=self.max_size + 1)

        # Compute new height / width
        height = image.size(1)
        width = image.size(2)
        if height < width:
            new_height = size
            new_width = size / height * width
        else:
            new_height = size / width * height
            new_width = size

        # Resize
        image_resized = torchvision.transforms.functional.resize(image, size=(new_height, new_width))

        return image_resized

class RandomClipCrop(nn.Module):

    def __init__(self, num_frames, rate_ratio):
        super().__init__()

        # F
        assert num_frames > 0
        self.num_frames = num_frames
        self.rate_ratio = rate_ratio

    def forward(self, vid, aud):

        # T
        vid_len = vid.size(1)

        if vid_len < self.num_frames:
            vid = torch.cat([vid, vid[:, :self.num_frames-vid_len]], dim=1)
            return vid, aud

        # T0 <= T - F
        start_frame = torch.randint(0, vid_len-self.num_frames+1, size=())

        # (C, T, H, W) -> (C, F, H, W)
        vid = vid[:, start_frame:start_frame+self.num_frames]

        # (C, Ta) -> (C, Fa)
        aud = aud[:, int(self.rate_ratio) * start_frame:int(self.rate_ratio) * (start_frame + self.num_frames)]

        return vid, aud

class TemporalStride(nn.Module):

    def __init__(self, temporal_stride):
        super().__init__()

        # S
        self.temporal_stride = temporal_stride

    def forward(self, vid):

        # (C, T, H, W) -> (C, T//S, H, W)
        vid = vid[:, ::self.temporal_stride]

        return vid

class ResizeRCNN(nn.Module):

    """

    images are scaled up such that the small side is = IMAGE_MIN_DIM, but ensuring that the
    scaling doesn't make the long side > IMAGE_MAX_DIM. Then the image is padded with zeros 
    to make it a square so multiple images can be put in one batch.
    
    """

    def __init__(self, min_size=600, max_size=1024):
        super().__init__()

        self.min_size = min_size
        self.max_size = max_size

    def forward(self, img):

        h, w = img.shape[1:]

        # Scale up but not down
        scale = max(1, self.min_size / min(h, w))

        # Does it exceed max dim?
        image_max = max(h, w)
        if round(image_max * scale) > self.max_size:
            scale = self.max_size / image_max

        # Resize image using bilinear interpolation
        if scale != 1:
            img = torchvision.transforms.functional.resize(img, size=(round(h * scale), round(w * scale)))

        # Padding
        h, w = img.shape[1:]
        top_pad = (self.max_size - h) // 2
        bottom_pad = self.max_size - h - top_pad
        left_pad = (self.max_size - w) // 2
        right_pad = self.max_size - w - left_pad
        img = torchvision.transforms.functional.pad(img, padding=(left_pad, top_pad, right_pad, bottom_pad))
        padding = (left_pad, top_pad, right_pad, bottom_pad)
        window = (left_pad, top_pad, left_pad + w , top_pad + h)

        return img, window, scale, padding

class Denormalize(nn.Module):

    def __init__(self, mean, std):
        super().__init__()

        self.mean = torch.tensor(mean).reshape(3, 1, 1)
        self.std = torch.tensor(std).reshape(3, 1, 1)

    def forward(self, x):

        x = x * self.std + self.mean

        return x

class NormalizeVideo(nn.Module):

    def __init__(self, mean, std):
        super().__init__()

        self.mean = torch.tensor(mean).reshape(3, 1, 1, 1)
        self.std = torch.tensor(std).reshape(3, 1, 1, 1)

    def forward(self, x):

        x = (x - self.mean) / self.std

        return x

class DenormalizeVideo(nn.Module):

    def __init__(self, mean, std):
        super().__init__()

        self.mean = torch.tensor(mean).reshape(3, 1, 1, 1)
        self.std = torch.tensor(std).reshape(3, 1, 1, 1)

    def forward(self, x):

        x = x * self.std + self.mean

        return x