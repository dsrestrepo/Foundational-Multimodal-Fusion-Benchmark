from src.pretrain_image import pretrain_model
import pandas as pd
import os

backbones = ['convnextv2_base', 'vit_base', 'dinov2_base']

batch_size = 64
epochs = 20

#path = '/gpfs/workdir/restrepoda/datasets/BRSET/brset/images/'
#path = '/gpfs/workdir/restrepoda/datasets/mBRSET/mbrset/images/'
path = '/gpfs/workdir/restrepoda/datasets/MIMIC/mimic/'

checkpoints = "/gpfs/workdir/restrepoda/checkpoints/"

#dataset = 'brset'
#dataset = 'mbrset'
dataset = 'mimic'

device = "cuda"
method = "dino" # contrastive # byol # dino

# BRSET
#files = pd.read_csv("/gpfs/workdir/restrepoda/datasets/BRSET/brset/labels.csv")
#files = files[files.split == 'train']
#files.file = path + files.image_id + ".jpg"

# mBRSET
#files = pd.read_csv("/gpfs/workdir/restrepoda/datasets/mBRSET/mbrset/labels.csv")
#files = files[files.split == 'train']
#files.file = path + files.file

# Mimic
files = pd.read_csv("/gpfs/workdir/restrepoda/datasets/MIMIC/mimic/labels.csv")
files = files[files.split == 'train']
files.file = path + files.path_preproc

image_files = files.file.tolist()

os.makedirs(checkpoints, exist_ok=True)

for backbone in backbones:
    print(f"Training {backbone} using {method}...")
    checkpoint_path = checkpoints + f"{backbone}-{dataset}-{method}.pt"
    pretrain_model(batch_size, path, dataset_name=dataset, backbone=backbone, directory=checkpoint_path, device=device, method=method, epochs=epochs, image_files=image_files)
