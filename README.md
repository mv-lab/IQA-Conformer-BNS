# NTIRE 2022 Perceptual Image Quality Assessment Challenge Track 1 Full-Reference

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

The PIPAL dataset should be downloded and extracted in the datasets/PIPAL folder:
PIPAL:
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

## Train IQA Conformer Model

Start training
```
python main.py -c configs/PIPAL/IQA_Conformer.py
```

Exponential Moving Average of last 10 checkpoints
```
python main.py -c configs/PIPAL/IQA_Conformer.py --mode swa --swa_epochs 21 30
```

Generate Submission files in callbacks/PIPAL/IQA_Conformer/eval
```
python main.py -c configs/PIPAL/IQA_Conformer.py --checkpoint swa-equal-21-30 --mode generation
```

