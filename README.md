# [CVPR NTIRE 2022] Conformer and Blind Noisy Students for Improved Image Quality Assessment.

Our approaches achieved top results on the [NTIRE 2022 Perceptual Image Quality Assessment Challenge](https://data.vision.ee.ethz.ch/cvl/ntire22/): our full-reference model was ranked **4th**, and our no-reference was ranked **3rd** among 70 participants.

[Read our paper here](https://arxiv.org/pdf/2204.12819.pdf)

> Generative models for image restoration, enhancement, and generation have significantly improved the quality of the generated images. Surprisingly, these models produce more pleasant images to the human eye than other methods, yet, they may get a lower perceptual quality score using traditional perceptual quality metrics such as PSNR or SSIM. Therefore, it is necessary to develop a quantitative metric to reflect the performance of new algorithms, which should be well-aligned with the person's mean opinion score (MOS). Learning-based approaches for perceptual image quality assessment (IQA) usually require both the distorted and reference image for measuring the perceptual quality accurately. However, commonly only the distorted or generated image is available. In this work, we explore the performance of transformer-based full-reference IQA models. We also propose a method for IQA based on semi-supervised knowledge distillation from full-reference teacher models into blind student models using noisy pseudo-labeled data. Our approaches achieved competitive results on the NTIRE 2022 Perceptual Image Quality Assessment Challenge: our full-reference model was ranked 4th, and our blind noisy student was ranked 3rd among 70 participants, each in their respective track. 

If you use ideas/results from this paper or code from this repo, don't forget to cite it :)

```

@inproceedings{conde2022ntire,
title={Conformer and Blind Noisy Students for Improved Image Quality Assessment}, 
author = {Conde, Marcos V. and Burchi, Maxime and Timofte, Radu},
booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
year={2022}
}
```

-------

## NTIRE 2022 Perceptual Image Quality Assessment Challenge Track 1 Full-Reference: IQA Conformer Network

### Installation

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

### Dataset

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

### Pretrained Models and Submissions

The repository contains two pretrained model checkpoints (IQA Conformer and IQA Transformer).
Both architectures were trained on the PIPAL training sets for 30 epochs (43479 gradient steps) and averaged via Stochastic Weight Averaging (SWA).

Model checkpoints are saved in callbacks/PIPAL/IQA_Conformer and callbacks/PIPAL/IQA_Transformer.
The submissions are stored in eval subfolders.

### Global Method Description

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

### Train IQA Transformer Model

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

### NTIRE 2022 IQA Challenge - PIPAL Full-Reference Performance: 

| Team       			|Main Score | PLCC | SRCC |
| :-------------------:	|:--------:	|:-----:|:----------:|
| THU1919Group  		|  1.651  | 0.828   | 0.822 |
| Netease OPDAI         |  1.642  | 0.827   | 0.815 |
| KS                    |  1.640  | 0.823   | 0.817 |
| Ours                  |  1.541  | 0.775   | 0.766 |
| Yahaha!               |  1.538  | 0.772   | 0.765 |
| debut kele            |  1.501  | 0.763   | 0.737 |
| Pico Zen              |  1.450  | 0.738   | 0.713 |
| Team Horizon          |  1.403  | 0.703   | 0.701 |

Pearson linear correlation coefficient (PLCC) and Spearman rank order correlation coefficient (SRCC)

Main Score is the sum of PLCC and SRCC, the higher the better. Teams ordered by rank int he challenge.


## References
[1] [Perceptual Image Quality Assessment with Transformers](https://openaccess.thecvf.com/content/CVPR2021W/NTIRE/papers/Cheon_Perceptual_Image_Quality_Assessment_With_Transformers_CVPRW_2021_paper.pdf) by Manri Cheon, Sung-Jun Yoon, Byungyeon Kang, and Junwoo Lee.
<br>

[2] [NTIRE 2021 Challenge on Perceptual Image Quality Assessment](https://openaccess.thecvf.com/content/CVPR2021W/NTIRE/papers/Gu_NTIRE_2021_Challenge_on_Perceptual_Image_Quality_Assessment_CVPRW_2021_paper.pdf) by Jinjin Gu, Haoming Cai, Chao Dong, Jimmy S. Ren, Yu Qiao, Shuhang Gu, Radu Timofte et al.
<br>

[3] [PIPAL: a Large-Scale Image Quality Assessment Dataset for Perceptual image Restoration](https://www.jasongt.com/projectpages/pipal.html) by Jinjin Gu, Haoming Cai, Haoyu Chen, Xiaoxing Ye, Jimmy Ren, Chao Dong.

-------

# Contacts
* Maxime Burchi [@burchim](https://github.com/burchim) | [maxime.burchi@gmail.com](mailto:maxime.burchi@gmail.com)
* Marcos Conde  [@mv-lab](https://github.com/mv-lab)   | [marcos.conde-osorio@uni-wuerzburg.de](mailto:marcos.conde-osorio@uni-wuerzburg.de)

