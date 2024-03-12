from src.embeddings import get_embeddings_df
import pandas as pd

# Foundational Models
dino_backbone = ['dinov2_small', 'dinov2_base', 'dinov2_large', 'dinov2_giant']

sam_backbone = ['sam_base', 'sam_large', 'sam_huge']

clip_backbone = ['clip_base', 'clip_large']

# ImageNet:

### Convnext
convnext_backbone = ['convnextv2_tiny', 'convnextv2_base', 'convnextv2_large'] + ['convnext_tiny', 'convnext_small', 'convnext_base', 'convnext_large']

### Swin Transformer
swin_transformer_backbone = ['swin_tiny', 'swin_small', 'swin_base']

### ViT
vit_backbone = ['vit_base', 'vit_large']

backbones = dino_backbone + clip_backbone + sam_backbone + convnext_backbone + swin_transformer_backbone + vit_backbone


batch_size = 32
path = 'datasets/joslin/images'
dataset = 'joslin'
backbone = 'dinov2_base'
out_dir = 'Embeddings'
device = "cuda"

get_embeddings_df(batch_size=batch_size, path=path, dataset_name=dataset, backbone=backbone, directory=out_dir, device = device)