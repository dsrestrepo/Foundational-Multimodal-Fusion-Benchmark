from .classifiers import process_labels

import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import torch
import os
import numpy as np

import warnings
warnings.filterwarnings("ignore")


# Custom Dataset class for PyTorch
class VQADataset(Dataset):
    """
    Custom PyTorch Dataset for VQA (Visual Question Answering).

    Args:
    - df (pd.DataFrame): The DataFrame containing the dataset.
    - text_cols (str): Column name containing the text.
    - image_cols (str): Column name containing the path to the images.
    - label_col (str): Column name containing labels.
    - mlb (sklearn.preprocessing.MultiLabelBinarizer): MultiLabelBinarizer object.
    - train_columns (list): List of columns from the training set.

    Attributes:
    - text_data (np.ndarray): Array of text data.
    - image_data (np.ndarray): Array of image data.
    - mlb (sklearn.preprocessing.MultiLabelBinarizer): MultiLabelBinarizer object.
    - train_columns (list): List of columns from the training set.
    - labels (np.ndarray): Array of one-hot encoded labels.

    Methods:
    - __len__(): Returns the length of the dataset.
    - __getitem__(idx): Returns a dictionary with 'text', 'image', and 'labels'.

    Example:
    dataset = VQADataset(df, text_cols=['text1', 'text2'], image_cols=['image1', 'image2'], label_col='answer', mlb=mlb, train_columns=train_columns)
    """
    def __init__(self, df, text_cols, image_cols, label_col, mlb, train_columns, tokenizer, max_len=50, shape=(224, 224), transform=None):
        # Text
        self.text_data = df[text_cols].values
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        # Images
        self.image_data = df[image_cols].values
        self.shape = shape
        self.transform = transform or transforms.Compose([
            transforms.Resize(self.shape),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Labels
        self.mlb = mlb
        self.train_columns = train_columns
        self.labels = process_labels(df, col=label_col, mlb=mlb, train_columns=train_columns).values
        #print(self.labels.shape)
        if len(self.labels.shape) == 1:
            self.labels = np.expand_dims(self.labels, axis=1)


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        
        # Images:
        img_path = self.image_data[idx]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        
        # Text:
        text = str(self.text_data[idx])
        txt = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_tensors='pt',
            truncation=True
        )
        
        return {
            'text': {
                'input_ids': txt['input_ids'].squeeze(),
                'attention_mask': txt['attention_mask'].squeeze()
            },
            'image': torch.FloatTensor(img),
            'labels': torch.FloatTensor(self.labels[idx])
        }

