# NTIRE 2022 Perceptual Image Quality Assessment Challenge Track 1 Full-Reference: IQA Conformer Network

## Installation

Create virtual env (optional)
```
python -m venv env
source env/bin/activate
```

Set up environment
```
pip install --upgrade pip
pip install -r requirements.txt
```

## Dataset

The [PIPAL dataset](https://www.jasongt.com/projectpages/pipal.html) should be downloded and extracted in the datasets/PIPAL folder:
```
datasets/PIPAL

    - Distortion1
        - A0001_00_00.bmp
        - ...

    - Distortion2
        - A0060_00_00.bmp
        - ...

    - Distortion3
        - A0121_00_00.bmp
        - ...

    - Distortion4
        - A0185_00_00.bmp
        - ...

    - NTIRE2022_FR_Testing_Dis
        - A0000_10_00.bmp
        - ...

    - NTIRE2022_FR_Testing_Ref
        - A0000.bmp
        - ...

    - Train_Label
        - A0001.txt
        - ...

    - Train_Ref
        - A0001.bmp
        - ...
```

## Pretrained Models and Submissions

The repository contains two pretrained model checkpoints (IQA Conformer and IQA Transformer).
Both architectures were trained on the PIPAL training sets for 30 epochs (43479 gradient steps) and averaged via Stochastic Weight Averaging (SWA).

Model checkpoints are saved in callbacks/PIPAL/IQA_Conformer and callbacks/PIPAL/IQA_Transformer.
The submissions are stored in eval subfolders.

## Global Method Description

IQA Transformer is a reimplementation of IQT-C "Perceptual Image Quality Assessment with Transformers" by Cheon et al.

Concerning IQA_Conformer, as done in Cheon et al., mixed5b, block35_2, block35_4, block35_6, block35_8 and block35_10 feature maps from an Inception-ResNet-v2 network pre-trained on ImageNet are concatenated for reference and distorted images generating f_ref and f_dist, respectively.
In order to obtain difference information between reference and distorted images,
a difference feature map, f_diff = f_ref - f_dist is also used.

Concatenated feature maps are then projected using a point-wise convolution but not flattened to preserve spatial information. We use a single Conformer block for both encoder and decoder. The model hyper-parameters are set as follow: L=1, D=128, H=4, D_feat=512, and D_head=128. The input image size of the backbone model is set to 192 × 192 × 3 which generates feature maps of size 21 x 21.

## Train IQA Conformer Model

Start training:
```
python main.py -c configs/PIPAL/IQA_Conformer.py
```

Exponential Moving Average of last 10 checkpoints:
```
python main.py -c configs/PIPAL/IQA_Conformer.py --mode swa --swa_epochs 21 30
```

Generate Submission files: 
```
python main.py -c configs/PIPAL/IQA_Conformer.py --checkpoint swa-equal-21-30 --mode generation
```
output.txt and readme.txt will be stored in callbacks/PIPAL/IQA_Conformer/eval

## Train IQA Transformer Model

Start training:
```
python main.py -c configs/PIPAL/IQA_Transformer.py
```

Exponential Moving Average of last 10 checkpoints:
```
python main.py -c configs/PIPAL/IQA_Transformer.py --mode swa --swa_epochs 21 30
```

Generate Submission files: 
```
python main.py -c configs/PIPAL/IQA_Transformer.py --checkpoint swa-equal-21-30 --mode generation
```
output.txt and readme.txt will be stored in callbacks/PIPAL/IQA_Transformer/eval

## PIPAL Performance: Pearson linear correlation coefficient (PLCC) and Spearman rank order correlation coefficient (SRCC)

| Model        			| val SRCC     	| val PLCC  | test SRCC | test PLCC |
| :-------------------:	|:--------:	|:-----:|:----------:|:------:|
| IQT [1] | 0.876 		| 0.865  | 0.790   | 0.799 |
| IQT (ours)| 0.7650		| 0.7897 | 0.7510  | 0.7571 |
| IQA Conformer| 0.7878 		| 0.8035 | 0.7659  | 0.7747 |

## References
[1] [Manri Cheon, Sung-Jun Yoon, Byungyeon Kang, and Junwoo Lee.	Perceptual Image Quality Assessment with Transformers.](https://openaccess.thecvf.com/content/CVPR2021W/NTIRE/papers/Cheon_Perceptual_Image_Quality_Assessment_With_Transformers_CVPRW_2021_paper.pdf)
<br><br>

## Author
* Maxime Burchi [@burchim](https://github.com/burchim)
* Contact: [maxime.burchi@gmail.com](mailto:maxime.burchi@gmail.com)

