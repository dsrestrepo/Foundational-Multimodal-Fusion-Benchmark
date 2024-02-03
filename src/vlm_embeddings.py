import argparse
import os
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from PIL import Image
import numpy as np
from torchvision.transforms import Resize, Compose, ToTensor
from transformers import AutoProcessor, LlavaForConditionalGeneration, Blip2Processor, Blip2ForConditionalGeneration
import torch
import torch.distributed as dist
from tqdm.auto import tqdm
from classifiers_base import preprocess_df

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# Setup distributed computing if available
if torch.distributed.is_available():
    dist.init_process_group(backend="gloo")
    local_rank = int(os.environ["LOCAL_RANK"])
    #device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

# Dataset and collate functions
class CustomDataset:
    def __init__(self, dataframe, image_col, text_col, image_path=None, processor=None, transform=None):
        self.dataframe = dataframe
        self.image_col = image_col
        self.text_col = text_col
        self.image_path = image_path
        self.processor = processor
        self.transform = transform or Compose([Resize((224, 224)), ToTensor()])

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        text = row[self.text_col]
        image_file = row[self.image_col] #os.path.join(self.image_path if self.image_path else '', row[self.image_col])
        image = Image.open(image_file).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return {"text": text, "image": image, "image_path": image_file}

def custom_collate_fn(batch, processor, model_type):
    if model_type.lower() == 'blip2':
        texts = [f"Question: {item['text']} Answer:" for item in batch]
    else:
        texts = [item["text"] for item in batch]
    images = [item["image"] for item in batch]
    image_paths = [item["image_path"] for item in batch]
    processed = processor(text=texts, images=images, return_tensors="pt", padding=True, truncation=True)
    processed['image_paths'] = image_paths
    return processed

# Functions for getting embeddings
def get_blip2_embeddings(dataframe, batch_size, image_col_name, text_col_name, image_path, output_dir, output_file, processor, model):
    # Initialize the dataset
    dataset = CustomDataset(dataframe, image_col_name, text_col_name, image_path, processor)

    # Set up a DistributedSampler to partition the dataset among the workers
    sampler = DistributedSampler(dataset, shuffle=False)

    # Initialize the DataLoader with the distributed sampler
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, collate_fn=lambda x: custom_collate_fn(x, processor, 'blip2'), num_workers=4)

    if dist.get_rank() == 0:
        progress_bar = tqdm(total=len(dataloader), desc="Processing batches")
        
    embeddings_list = []
    image_paths_list = []

    for batch in tqdm(dataloader, desc="Processing batches"):
        #batch_inputs = {k: v.to(device) for k, v in batch.items() if k != 'image_paths'}
        batch_inputs = {k: v for k, v in batch.items() if k != 'image_paths'}
        image_paths_list.extend(batch['image_paths'])

        with torch.no_grad():
            outputs = model(**batch_inputs)
        
        # Extract embeddings from the model's output
        embeddings = outputs['qformer_outputs']['pooler_output'].cpu().numpy()
        embeddings_list.append(embeddings)
        
        # Update the progress on the master process
        if dist.get_rank() == 0:
            progress_bar.update(1)

    if dist.get_rank() == 0:
        progress_bar.close()

    # Gather all embeddings and image paths from all processes
    all_embeddings = [None] * dist.get_world_size()
    all_image_paths = [None] * dist.get_world_size()
    dist.all_gather_object(all_embeddings, embeddings_list)
    dist.all_gather_object(all_image_paths, image_paths_list)

    # Concatenate embeddings and image paths from all processes
    if dist.get_rank() == 0:  # Only the main process will save the output
        embeddings = np.concatenate(all_embeddings, axis=0)
        image_paths = sum(all_image_paths, [])  # Flatten the list of lists

        embeddings_df = pd.DataFrame(embeddings, columns=[f'embedding_{i}' for i in range(embeddings.shape[1])])
        embeddings_df['image_path'] = image_paths

        # Merge the original dataframe with the embeddings dataframe on the image path
        concatenated_df = pd.merge(dataframe, embeddings_df, on='image_path', how='left')

        # Save the final DataFrame to a CSV file
        output_path = os.path.join(output_dir, output_file)
        concatenated_df.to_csv(output_path, index=False)


def get_llava_embeddings(dataframe, batch_size, image_col_name, text_col_name, image_path, output_dir, output_file, processor, model):
   # Initialize the dataset
    dataset = CustomDataset(dataframe, image_col_name, text_col_name, image_path, processor)
    
    # Set up a DistributedSampler to partition the dataset among the workers
    sampler = DistributedSampler(dataset, shuffle=False)

    # Initialize the DataLoader with the distributed sampler
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, collate_fn=lambda x: custom_collate_fn(x, processor, 'llava'), num_workers=4)
    
    if dist.get_rank() == 0:
        progress_bar = tqdm(total=len(dataloader), desc="Processing batches")

    embeddings_list = []
    image_paths_list = []

    for batch in tqdm(dataloader, desc="Processing batches"):
        #batch_inputs = {k: v.to(device) for k, v in batch.items() if k != 'image_paths'}
        batch_inputs = {k: v for k, v in batch.items() if k != 'image_paths'}
        image_paths_list.extend(batch['image_paths'])
        
        with torch.no_grad():
            outputs = model(**batch_inputs, output_hidden_states=True)
        
        # Extract embeddings from the last hidden state and perform mean pooling
        last_hidden_state = outputs.hidden_states[-1]  # Shape: (batch_size, sequence_length, hidden_size)
        pooled_embeddings = last_hidden_state.mean(dim=1).cpu().numpy()  # Shape: (batch_size, hidden_size)
        
        embeddings_list.append(pooled_embeddings)

    # Gather all embeddings and image paths from all processes
    all_embeddings = [None] * dist.get_world_size()
    all_image_paths = [None] * dist.get_world_size()
    dist.all_gather_object(all_embeddings, embeddings_list)
    dist.all_gather_object(all_image_paths, image_paths_list)

    # Concatenate embeddings and image paths from all processes
    if dist.get_rank() == 0:  # Only the main process will save the output
        embeddings = np.concatenate(all_embeddings, axis=0)
        image_paths = sum(all_image_paths, [])  # Flatten the list of lists

        embeddings_df = pd.DataFrame(embeddings, columns=[f'embedding_{i}' for i in range(embeddings.shape[1])])
        embeddings_df['image_path'] = image_paths

        # Merge the original dataframe with the embeddings dataframe on the image path
        concatenated_df = pd.merge(dataframe, embeddings_df, on='image_path', how='left')

        # Save the final DataFrame to a CSV file
        output_path = os.path.join(output_dir, output_file)
        concatenated_df.to_csv(output_path, index=False)


def main():
    parser = argparse.ArgumentParser(description="Extract embeddings from VLMs.")
    parser.add_argument("--classifier", type=str, required=True, help="Classifier to use: LLAVA or BLIP2")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for processing")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset")
    parser.add_argument("--image_col", type=str, required=True, help="Name of the image column")
    parser.add_argument("--text_col", type=str, required=True, help="Name of the text column")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for embeddings")
    parser.add_argument("--output_file", type=str, required=True, help="Output file name for embeddings")
    parser.add_argument("--image_dir", type=str, required=False, default='images', help="Name of directory with the images to the dataset")    
    parser.add_argument("--labels", type=str, required=False, default='labels.csv', help="Name of the file with the labels ans text")
    args = parser.parse_args()

    # Load dataframe
    labels_path = os.path.join(args.dataset_path, args.labels)
    images_path = os.path.join(args.dataset_path, args.image_dir)
    df = pd.read_csv(labels_path)
    df = preprocess_df(df=pd.read_csv(labels_path), image_columns=args.image_col, images_path=images_path)


    # Initialize model and processor
    if args.classifier.lower() == 'blip2':
        processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")#.to(device)
        
        if torch.distributed.is_available() and not isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model = torch.nn.parallel.DistributedDataParallel(model)#, device_ids=[local_rank], output_device=local_rank)

        get_embeddings = get_blip2_embeddings
    elif args.classifier.lower() == 'llava':
        processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
        model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")#.to(device)
        
        if torch.distributed.is_available() and not isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model = torch.nn.parallel.DistributedDataParallel(model)#, device_ids=[local_rank], output_device=local_rank)
        get_embeddings = get_llava_embeddings
    else:
        raise ValueError("Unsupported classifier. Choose either LLAVA or BLIP2.")

    # Get embeddings
    print('Starting...')
    get_embeddings(df, args.batch_size, args.image_col, args.text_col, args.image_col, args.output_dir, args.output_file, processor, model)

if __name__ == "__main__":
    main()
