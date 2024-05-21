# Bayesian-Guided Label Mapping for Visual Reprogramming
This is the implementation of our paper submitted for NIPS2024.

## Installation
        conda create -n reprogram
        conda activate reprogram
        pip install -r requirements.txt

## Dataset Preparation
To implement the results, please follow Appendix D to download the dataset first, and modify the 'data_path' in cfg.py.

## Training
### Bayesian-Guided Label Mapping (BLM)
        python train_vm.py --dataset sun397 --mapping blm --seed 0
        python train_vlm.py --dataset sun397 --mapping blm --seed 0

### Improved Bayesian-Guided Label Mapping (BLM+)
        python train_vm.py --dataset sun397 --mapping blmpp --seed 0
        python train_vlm.py --dataset sun397 --mapping blmpp --seed 0

