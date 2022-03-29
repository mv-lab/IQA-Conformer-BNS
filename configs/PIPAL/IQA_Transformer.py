import sys
sys.path.append("../../")

# Neural Nets
import nnet

# PyTorch
import torch
import torch.nn as nn
import torchvision

# Other
import os
from tqdm import tqdm
import glob
from PIL import Image
import time
import pretrainedmodels

os.environ["TORCH_HOME"] = "pretrainedmodels"

mos_mean=1448.9595
mos_std=121.5351
dim_concat = 1920
dim_model = 128
ff_ratio = 8
mean=[0.5, 0.5, 0.5]
std=[0.5, 0.5, 0.5]

class CNN_Model(nnet.Regressor):

    def __init__(self):
        super(CNN_Model, self).__init__()

        # Training backbone
        train_backbone = False

        # Backbone modules
        backbone_modules = list(pretrainedmodels.inceptionresnetv2().modules())

        # Stem
        self.mixed5b = nn.Sequential(backbone_modules[1], backbone_modules[5], backbone_modules[9], backbone_modules[13], backbone_modules[14], backbone_modules[18], backbone_modules[22], backbone_modules[23])

        # Block 1
        self.block35_2 = nn.Sequential(backbone_modules[57], backbone_modules[86])

        # Block 2
        self.block35_4 = nn.Sequential(backbone_modules[115], backbone_modules[144])

        # Block 3
        self.block35_6 = nn.Sequential(backbone_modules[173], backbone_modules[202])

        # Block 4
        self.block35_8 = nn.Sequential(backbone_modules[231], backbone_modules[260])

        # Block 5
        self.block35_10 = nn.Sequential(backbone_modules[289], backbone_modules[318])

        # Set require grad
        self.set_require_grad(self.mixed5b, train_backbone)
        self.set_require_grad(self.block35_2, train_backbone)
        self.set_require_grad(self.block35_4, train_backbone)
        self.set_require_grad(self.block35_6, train_backbone)
        self.set_require_grad(self.block35_8, train_backbone)
        self.set_require_grad(self.block35_10, train_backbone)

        # Proj
        self.conv_proj = nnet.Conv2d(in_channels=dim_concat, out_channels=dim_model, kernel_size=(1, 1))

        H, W = 21, 21

        # Emb
        self.emb_q = nnet.Embedding(num_embeddings=2, embedding_dim=dim_model)
        self.emb_p = nnet.Embedding(num_embeddings=H * W + 1, embedding_dim=dim_model)

        # Encoder
        self.encoder = nnet.Transformer(
            dim_model=dim_model,
            num_blocks=1,
            ff_ratio=ff_ratio
        )

        # Decoder
        self.decoder = nnet.CrossTransformer(
            dim_model=dim_model,
            num_blocks=1,
            ff_ratio=ff_ratio
        )

        # Head
        self.head = nn.Sequential(
            nnet.Linear(in_features=dim_model, out_features=dim_model),
            nnet.ReLU(),
            nnet.Linear(in_features=dim_model, out_features=1),
        )

        self.trainable = [self.conv_proj, self.emb_q, self.emb_p, self.encoder, self.decoder, self.head]
        self.frozen = [self.mixed5b, self.block35_2, self.block35_4, self.block35_6, self.block35_8, self.block35_10]

    def summary(self, show_dict=False):
        super(CNN_Model, self).summary(show_dict=False)

        print("Frozen Parameters: {:,}".format(self.num_params(self.frozen)))
        if show_dict:
            self.show_dict(self.frozen)

        print("Trainable Parameters: {:,}".format(self.num_params(self.trainable)))
        if show_dict:
            self.show_dict(self.trainable)

    def backbone(self, x):

        self.mixed5b.eval()
        self.block35_2.eval()
        self.block35_4.eval()
        self.block35_6.eval()
        self.block35_8.eval()
        self.block35_10.eval()

        x1 = self.mixed5b(x)
        x2 = self.block35_2(x1)
        x3 = self.block35_4(x2)
        x4 = self.block35_6(x3)
        x5 = self.block35_8(x4)
        x6 = self.block35_10(x5)

        return torch.cat([x1, x2, x3, x4, x5, x6], dim=1)

    def forward(self, x):

        x_ref, x_dis = x

        with torch.no_grad():
            x_ref = self.backbone(x_ref)
            x_dis = self.backbone(x_dis)

        x_diff = x_ref - x_dis

        x_ref = self.conv_proj(x_ref)
        x_diff = self.conv_proj(x_diff)
        
        x_ref = x_ref.flatten(2, -1).transpose(1, 2)
        x_diff = x_diff.flatten(2, -1).transpose(1, 2)

        x_ref = torch.cat([self.emb_q(torch.zeros(x_ref.size(0), 1, device=x_ref.device, dtype=torch.int)), x_ref], dim=1) + self.emb_p(torch.arange(1 + x_ref.size(1), device=x_ref.device).unsqueeze(dim=0))
        x_diff = torch.cat([self.emb_q(torch.ones(x_diff.size(0), 1, device=x_diff.device, dtype=torch.int)), x_diff], dim=1) + self.emb_p(torch.arange(1 + x_diff.size(1), device=x_diff.device).unsqueeze(dim=0))

        x_enc = self.encoder(x_diff)
        x_dec = self.decoder(x_ref, x_enc)

        x = self.head(x_dec[:, 0])

        return x

    def generate(self, dataset, saving_path=None):

        # Eval mode
        self.eval()

        # Model Device
        self.device = next(self.parameters()).device

        # Create Saving Path
        if saving_path != None:
            if not os.path.isdir(saving_path):
                os.makedirs(saving_path)

        # Init
        dir_dis = "NTIRE2022_FR_Testing_Dis"
        dir_ref = "NTIRE2022_FR_Testing_Ref"
        dis = tqdm(sorted(glob.glob(os.path.join("datasets", "PIPAL", dir_dis, "*.bmp"))))

        transforms = torchvision.transforms.Compose([
            torchvision.transforms.Lambda(lambda x: x.convert("RGB") if x.mode != "RGB" else x),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=mean, std=std),
            torchvision.transforms.TenCrop(size=(192, 192)),
            torchvision.transforms.Lambda(lambda crops: torch.stack([crop for crop in crops], dim=0))
        ])

        with open(os.path.join(saving_path, "output.txt"), "w") as f:

            start = time.time()

            # Epoch training loop
            for step, path in enumerate(dis):

                # Create Path Ref
                path_ref = path.replace(dir_dis, dir_ref)
                path_ref = "/".join(path.replace(dir_dis, dir_ref).split("/")[:-1]) + "/" + path.split("/")[-1][:5] + ".bmp"

                # Open Images
                img_ref = transforms(Image.open(path_ref))
                img_ref = self.transfer_to_device(img_ref)
                img = transforms(Image.open(path))
                img = self.transfer_to_device(img)

                # Generate Scores
                pred = (self.forward([img_ref, img]) * mos_std + mos_mean).mean()

                # Write
                f.write("{},{:.6f}".format(path.split("/")[-1], pred.item()))
                dis.set_description("{}, {}, {:.6f}".format(path.split("/")[-1], path_ref.split("/")[-1], pred.item()))

                # Stop
                if step < len(dis)-1:
                    f.write("\n")

            total_time = time.time() - start
            igm_time = total_time / len(dis)

        with open(os.path.join(saving_path, "readme.txt"), "w") as f:

            description = "runtime per image [s] : {:.2f}\n".format(igm_time)
            description += "CPU[1] / GPU[0] : {}\n".format("0" if torch.cuda.is_available() else "1")
            description += "Extra Data [1] / No Extra Data [0] : 1\n"
            description += "Other description : Solution based on an inception resnet v2 network pretrained on ImageNet and a Transformer post Network. Distorted images MOS are regressed using a Mean Squared Error loss. The method was trained on the PIPAL dataset training subset.\n"
            description += "Full-Reference [1] / Non-Reference [0] : 1"

            f.write(description)
            print(description)

def decode(outputs, from_logits=None):
    return outputs * mos_std + mos_mean

model = CNN_Model()
model.compile(
    optimizer=nnet.Adam(params=model.parameters(), lr=0.0001),
    decoders=decode,
    metrics=[[nnet.MeanAbsoluteDist(), nnet.PLCC(), nnet.SROCC(), nnet.PearsonSpearman()]],
    collate_fn=nnet.CollateList(inputs_axis=[0, 1], targets_axis=[2])
)


training_dataset = nnet.datasets.PIPAL(
    root="datasets/",
    subset="train",
    augments=torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(size=(192, 192)),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomRotation(180)
    ]),
    img_mean=mean,
    img_std=std
)

# Training Params
epochs = 100
batch_size = 16
accumulated_steps = 1
mixed_precision = True
callback_path = "callbacks/PIPAL/IQA_Transformer/"
generation_path = "callbacks/PIPAL/IQA_Transformer/eval/"
