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

The PIPAL dataset should be downloded/extracted and move to datasets/PIPAL:
PIPAL:
    - Distortion1
    - Distortion2
    - Distortion3
    - Distortion4
    - NTIRE2022_FR_Testing_Dis
    - NTIRE2022_FR_Testing_Ref
    - Train_Label
    - Train_Ref

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

