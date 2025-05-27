from src.embeddings import get_embeddings_df
import pandas as pd

# Foundational Models
#dino_backbone = ['dinov2_small', 'dinov2_base', 'dinov2_large', 'dinov2_giant']

### Convnext
#convnext_backbone = ['convnextv2_tiny', 'convnextv2_base', 'convnextv2_large'] + ['convnext_tiny', 'convnext_small', 'convnext_base', 'convnext_large']

### ViT
#vit_backbone = ['vit_base', 'vit_large']

backbones = ['dinov2_base', 'convnextv2_base', 'vit_base'] # 'dinov2_base'

batch_size = 32
path = '/gpfs/workdir/restrepoda/datasets/mBRSET/mbrset/images'
dataset = 'mbrset'
out_dir = 'Embeddings'
device = "cuda"

for backbone in backbones:
    get_embeddings_df(batch_size=batch_size, path=path, dataset_name=dataset, backbone=backbone, directory=out_dir, device = device)