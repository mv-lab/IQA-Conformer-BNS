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
import torchvision
import torchaudio
import torchtext
from torchvision.datasets.utils import (
    download_and_extract_archive,
    download_file_from_google_drive,
    download_url,
    extract_archive
)

# Other
import os
import glob
import random
from tqdm import tqdm
import sentencepiece as spm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Transforms
from nnet.transforms import (
    RandomResizeShorterSide,
    RandomClipCrop,
    TemporalStride,
    ResizeRCNN,
    Denormalize,
    NormalizeVideo,
    DenormalizeVideo
)

try:
    import sounddevice
except Exception as e:
    pass
    #print(e)

###############################################################################
# Transforms
###############################################################################

# PIPAL Dataset: Perceptual Image Processing ALgorithms
# https://www.jasongt.com/projectpages/pipal.html
# Images 3x288x288
# 250 high quality reference images 
# 40 distortion types, including the output of GAN-based algorithms
# 29,000 distortion images
# 1,130,000 human ratings, summarized by Mean Opinion Score (MOS) 
# in elo rating for each distortion image
# 200 reference training samples / 23200 distorted training samples
# 25 reference validation samples / 1650 distorted validation samples
# 25 reference test samples / 1650 distorted test samples
class PIPAL(torch.utils.data.Dataset):

    def __init__(self, root, subset="train", augments=None, mos_mean=1448.9595, mos_std=121.5351, img_mean=[0.485, 0.456, 0.406], img_std=[0.229, 0.224, 0.225]):

        assert subset in ["train", "valid", "test", "train_0-80", "train_80-100", "train_0-90", "train_90-100"]

        # Distortion Images
        self.dis = sorted(glob.glob(os.path.join(root, "PIPAL", "Distortion_*", "*.bmp")))

        # Scores
        score_files  = glob.glob(os.path.join(root, "PIPAL", "Train_Label", "*.txt"))
        score_dict = {}
        for file in score_files:
            for line in open(file).readlines():
                line0, line1 = line.split(",")
                score_dict[line0] = float(line1)

        # Reference Images and scores
        self.ref = []
        self.scores = []
        for path in self.dis:
            self.ref.append(os.path.join(root, "PIPAL", "Train_Ref", path.split("/")[-1].split("_")[0] + ".bmp"))
            self.scores.append(score_dict[path.split("/")[-1]])

        assert len(self.scores) == len(self.ref) == len(self.dis)

        # Transforms
        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.Lambda(lambda x: x.convert("RGB") if x.mode != "RGB" else x),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=img_mean, std=img_std)
        ])

        # Denormalize
        self.denorm = Denormalize(mean=img_mean, std=img_std)

        if subset == "train_0-80":
            self.dis = self.dis[:int(0.8*len(self.dis))]
            self.ref = self.ref[:int(0.8*len(self.ref))]
            self.scores = self.scores[:int(0.8*len(self.scores))]
        elif subset == "train_0-90":
            self.dis = self.dis[:int(0.9*len(self.dis))]
            self.ref = self.ref[:int(0.9*len(self.ref))]
            self.scores = self.scores[:int(0.9*len(self.scores))]
        elif subset == "train_80-100":
            self.dis = self.dis[int(0.8*len(self.dis)):]
            self.ref = self.ref[int(0.8*len(self.ref)):]
            self.scores = self.scores[int(0.8*len(self.scores)):]
        elif subset == "train_90-100":
            self.dis = self.dis[int(0.9*len(self.dis)):]
            self.ref = self.ref[int(0.9*len(self.ref)):]
            self.scores = self.scores[int(0.9*len(self.scores)):]

        # Params
        self.augments = augments
        self.mos_mean = mos_mean
        self.mos_std = mos_std

        # Print Real Scores Mean / Std
        #print("MOS mean {} / MOS std {}".format(torch.tensor(self.scores).mean(), torch.tensor(self.scores).std()))

    def download(self):
        pass

    def __getitem__(self, n):

        ref = self.transforms(Image.open(self.ref[n]))
        dis = self.transforms(Image.open(self.dis[n]))
        elo_rating = torch.tensor([(self.scores[n] - self.mos_mean) / self.mos_std])

        if self.augments != None:
            ref_dis = torch.cat([ref, dis], dim=0)
            ref_dis = self.augments(ref_dis)
            ref, dis = ref_dis[:3], ref_dis[3:]

        return ref, dis, elo_rating
        
    def __len__(self):

        return len(self.dis)

    def show(self, num_samples=10):

        # Shuffle indices
        indices = torch.arange(self.__len__()).numpy()
        np.random.shuffle(indices)

        # Show Samples
        for i in indices[:num_samples]:
            
            img_ref, img_dis, score = self.__getitem__(i)

            print("Score:", score)
            print(img_ref.size())
            print(img_ref)
            print(img_dis.size())
            print(img_dis)

            # Show
            plt.figure(figsize=(10, 10))

            plt.subplot(1, 2, 1)
            plt.imshow(self.denorm(img_ref).permute(1, 2, 0))
            plt.title("Image Ref / sample: {}".format(i) + "\n" + "min: {:.2f}, max:{:.2f}, mean: {:.2f}".format(img_ref.min(), img_ref.max(), img_ref.mean()))
                
            plt.subplot(1, 2, 2)
            plt.imshow(self.denorm(img_dis).permute(1, 2, 0))
            plt.title("Image Dis" + "\n" "score norm {:.2f} / score {:.2f}".format(score.item(), score.item() * self.mos_std + self.mos_mean))
            
            plt.show()