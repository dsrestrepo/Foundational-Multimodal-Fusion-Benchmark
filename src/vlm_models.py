from transformers import CLIPModel, CLIPProcessor
from transformers import AutoProcessor, LlavaForConditionalGeneration
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
import torch.nn as nn
import subprocess
import os
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
from tqdm.auto import tqdm

from torchvision.transforms import Resize, Compose, ToTensor

device = "cuda" if torch.cuda.is_available() else "cpu"

import warnings
warnings.filterwarnings("ignore")


class CLIPDataset(Dataset):
    def __init__(self, dataframe, image_col, text_col, image_path=None, processor=None):
        self.dataframe = dataframe
        self.image_col = image_col
        self.text_col = text_col
        self.image_path = image_path
        self.processor = processor

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        text = row[self.text_col]
        image_file = os.path.join(self.image_path if self.image_path else '', row[self.image_col])
        image = Image.open(image_file).convert("RGB")
        return {"text": text, "image": image}

def custom_collate_fn(batch, processor):
    texts = [item["text"] for item in batch]
    images = [item["image"] for item in batch]
    processed = processor(text=texts, images=images, return_tensors="pt", padding=True, truncation=True)
    return processed

class CLIP:
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def get_embeddings(self, dataframe, batch_size, image_col_name, text_col_name, image_path=None, output_dir='vlm_embeddings', output_file='clip_embeddings.csv', device="cuda"):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            # Check if the device string is valid
        if device not in ['cuda', 'cpu']:
            raise ValueError("Invalid device name. Please provide 'cuda' or 'cpu'.")
        
        device = torch.device(device)
        dataset = CLIPDataset(dataframe, image_col_name, text_col_name, image_path, self.processor)
        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=lambda x: custom_collate_fn(x, self.processor))

        # Initialize lists to store embeddings
        image_embeddings_list = []
        text_embeddings_list = []

        # Process batches with a progress bar
        for batch in tqdm(dataloader, desc="Processing batches"):
            with torch.no_grad():
                self.model.to(device)
                batch = batch.to(device)
                outputs = self.model(**batch)
            image_embeddings_list.append(outputs.image_embeds.cpu().numpy())
            text_embeddings_list.append(outputs.text_embeds.cpu().numpy())

        # Concatenate all embeddings
        image_embeddings = np.concatenate(image_embeddings_list, axis=0)
        text_embeddings = np.concatenate(text_embeddings_list, axis=0)

        # Create dataframes from embeddings
        image_embeddings_df = pd.DataFrame(image_embeddings, columns=[f'image_embedding_{i}' for i in range(image_embeddings.shape[1])])
        text_embeddings_df = pd.DataFrame(text_embeddings, columns=[f'text_embedding_{i}' for i in range(text_embeddings.shape[1])])

        # Concatenate the original dataframe with the embeddings dataframes
        concatenated_df = pd.concat([dataframe.reset_index(drop=True), image_embeddings_df, text_embeddings_df], axis=1)

        output_path = os.path.join(output_dir, output_file)
        concatenated_df.to_csv(output_path, index=False)

        return concatenated_df


class BLIP2Dataset(Dataset):
    def __init__(self, dataframe, image_col, text_col, image_path=None, processor=None):
        self.dataframe = dataframe
        self.image_col = image_col
        self.text_col = text_col
        self.image_path = image_path
        self.processor = processor
        self.transform = Compose([Resize((224, 224)), ToTensor()])

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        text = row[self.text_col]
        image_file = os.path.join(self.image_path if self.image_path else '', row[self.image_col])
        image = Image.open(image_file).convert("RGB")
        image = self.transform(image)
        return {"text": text, "image": image}

def custom_blip2_collate_fn(batch, processor):
    texts = [f"Question: {item['text']} Answer:" for item in batch]
    images = [item["image"] for item in batch]
    processed = processor(images=images, text=texts, return_tensors="pt", padding=True, truncation=True)
    return processed

class BLIP2:
    def __init__(self, model_name="Salesforce/blip2-opt-2.7b"):
        self.processor = Blip2Processor.from_pretrained(model_name)
        self.model = Blip2ForConditionalGeneration.from_pretrained(model_name).to(device)

    def get_embeddings(self, dataframe, batch_size, image_col_name, text_col_name, image_path=None, output_dir='vlm_embeddings', output_file='blip2_embeddings.csv', num_workers=24):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        dataset = BLIP2Dataset(dataframe, image_col_name, text_col_name, image_path, self.processor)
        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=lambda x: custom_blip2_collate_fn(x, self.processor), num_workers=num_workers)

        # Initialize list to store qformer outputs
        qformer_outputs_list = []

        # Process batches with a progress bar
        i=0
        for batch in tqdm(dataloader, desc="Processing batches"):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = self.model(**batch)
            qformer_outputs = outputs['qformer_outputs']['pooler_output'].cpu().numpy()
            qformer_outputs_list.append(qformer_outputs)
            if i % 100 == 0:
                print(f'Batch {i}')
                i+=1

        # Concatenate all qformer outputs
        qformer_outputs = np.concatenate(qformer_outputs_list, axis=0)

        # Create dataframe from qformer outputs
        qformer_outputs_df = pd.DataFrame(qformer_outputs, columns=[f'qformer_{i}' for i in range(qformer_outputs.shape[1])])

        # Concatenate the original dataframe with the qformer outputs dataframe
        concatenated_df = pd.concat([dataframe.reset_index(drop=True), qformer_outputs_df], axis=1)

        output_path = os.path.join(output_dir, output_file)
        concatenated_df.to_csv(output_path, index=False)

        return concatenated_df


class LlavaDataset(Dataset):
    def __init__(self, dataframe, image_col, text_col, image_path=None, processor=None):
        self.dataframe = dataframe
        self.image_col = image_col
        self.text_col = text_col
        self.image_path = image_path
        self.processor = processor
        self.transform = Compose([Resize((224, 224)), ToTensor()])

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        text = f"<image>\nUSER: {row[self.text_col]}\nASSISTANT:"
        image_file = os.path.join(self.image_path if self.image_path else '', row[self.image_col])
        image = Image.open(image_file).convert("RGB")
        image = self.transform(image)
        return {"text": text, "image": image}

def custom_llava_collate_fn(batch, processor):
    texts = [item["text"] for item in batch]
    images = [item["image"] for item in batch]
    processed = processor(text=texts, images=images, return_tensors="pt", padding=True, truncation=True)
    return processed

class LLAVA:
    def __init__(self, model_name="llava-hf/llava-1.5-7b-hf"):
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = LlavaForConditionalGeneration.from_pretrained(model_name).to(device)

    def get_embeddings(self, dataframe, batch_size, image_col_name, text_col_name, image_path=None, output_dir='vlm_embeddings', output_file='llava_embeddings.csv', num_workers=24):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        dataset = LlavaDataset(dataframe, image_col_name, text_col_name, image_path, self.processor)
        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=lambda x: custom_llava_collate_fn(x, self.processor), num_workers=num_workers)

        embeddings_list = []
        i = 0
        for batch in tqdm(dataloader, desc="Processing batches"):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = self.model(**batch, output_hidden_states=True)
            last_hidden_states = outputs['hidden_states'][-1]  # Shape: (batch, seq_len, dim)
            pooled_outputs = last_hidden_states.mean(dim=1).cpu().numpy()  # Mean pooling
            embeddings_list.append(pooled_outputs)
            
            if i % 100 == 0:
                print(f'Batch {i}')
                i+=1

        embeddings = np.concatenate(embeddings_list, axis=0)

        embeddings_df = pd.DataFrame(embeddings, columns=[f'llava_{i}' for i in range(embeddings.shape[1])])

        concatenated_df = pd.concat([dataframe.reset_index(drop=True), embeddings_df], axis=1)

        output_path = os.path.join(output_dir, output_file)
        concatenated_df.to_csv(output_path, index=False)

        return concatenated_df
