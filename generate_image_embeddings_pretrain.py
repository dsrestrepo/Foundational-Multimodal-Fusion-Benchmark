from src.embeddings import get_embeddings_df
import os


for dataset in ['brset', 'mbrset']:
    # Adjust the image path based on the dataset
    if dataset == 'brset':
        img_path = '/gpfs/workdir/restrepoda/datasets/BRSET/brset/images/'
    else:
        img_path = '/gpfs/workdir/restrepoda/datasets/mBRSET/mbrset/images/'

    for method in ['contrastive', 'byol', 'dino']:
        for backbone in ['convnextv2_base', 'vit_base', 'dinov2_base']:
            # Build the checkpoint path based on how you saved it:
            checkpoint_path = f"/gpfs/workdir/restrepoda/checkpoints/{backbone}-{dataset}-{method}.pt"
            get_embeddings_df(
                batch_size=32,
                path=img_path,
                dataset_name=dataset,
                backbone=backbone,
                directory='Embeddings',
                device="cuda",
                checkpoint_path=checkpoint_path
            )
