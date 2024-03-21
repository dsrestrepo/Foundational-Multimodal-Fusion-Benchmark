import os

import pandas as pd
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import MultiLabelBinarizer

# Function to train classic ML model
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset

# Metrics
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import RocCurveDisplay

from transformers import ViTModel, ViTConfig
from transformers import BertModel, BertConfig

from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing

import time

import warnings
# Suppress all warnings
warnings.filterwarnings("ignore")

######### Datasets Preparation #########
def preprocess_df(df, image_columns, images_path):
    # Function to check if an image can be opened
    def is_valid_image(img_path):
        img_path = os.path.join(images_path, img_path)
        # img_path = str(int(img_path)) + ".jpeg"
        # print(img_path)
        
        try:
            Image.open(img_path).convert("RGB")
            # print(img_path)
            return True
        except:
            
            print("invalid path!")
            return False

    # Function to correct image paths without extensions
    def correct_image_path(img_path):
        if type(img_path) != str:
            img_path = str(img_path)
            # if len(img_path) < 12:
            #     img_path = '0' * (12 - len(img_path)) + img_path
    
        full_img_path = os.path.join(images_path, img_path)
        img_path, file_name = os.path.split(full_img_path)

        if '.' not in file_name:
            # Try to find the correct extension in the directory
            for file in os.listdir(img_path):
                if file.split('.')[0] == file_name:
                    return os.path.join(images_path, file) 
        return full_img_path

    # Correct image paths if necessary
    df[image_columns] = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(correct_image_path)(img_path) for img_path in tqdm(df[image_columns]))

    # Filter out rows with images that cannot be opened
    valid_mask = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(is_valid_image)(img_path) for img_path in tqdm(df[image_columns]))
    df = df[valid_mask]

    # Correct image paths if necessary
    #df[image_columns] = df[image_columns].apply(correct_image_path)

    # Filter out rows with images that cannot be opened
    #df = df[df[image_columns].apply(is_valid_image)]

    return df
    
    
    
######### Models and Evaluation #########
class VisionModel(torch.nn.Module):
    """
    A PyTorch module for loading and using foundational computer vision models.

    This module allows you to load and use various ViT configurations.

    Parameters:
    - config (dict): The configuration of the ViT model.
    - mode (str, optional): The mode of the model, 'eval' for evaluation or 'fine_tune' for fine-tuning. Default is 'eval'.

    Methods:
    - forward(x): Forward pass to obtain features from input data.

    Example Usage:
    ```python
    config = {"attention_probs_dropout_prob": 0.0,
          "encoder_stride": 16,
          "hidden_act": "gelu",
          "hidden_dropout_prob": 0.0,
          "hidden_size": 768,
          "image_size": 224,
          "initializer_range": 0.02,
          "intermediate_size": 3072,
          "layer_norm_eps": 1e-12,
          "model_type": "vit",
          "num_attention_heads": 12,
          "num_channels": 3,
          "num_hidden_layers": 12,
          "patch_size": 16,
          "qkv_bias": true,
          "transformers_version": "4.35.0"}
    cv_model = VisionModel(config)
    features = cv_model(input_data)
    ```

    Note:
    - This module provides access to a ViT architecture.
    - It allows for both evaluation and fine-tuning modes.

    Dependencies:
    - PyTorch
    - Hugging Face Transformers (for ViT)

    For more information on specific models, refer to the respective model's documentation.
    """
    
    def __init__(self, config=None, mode='fine_tune', dropout_rate=0.2):
        """
        Initialize the FoundationalCVModel module.

        Args:
        - backbone (str): The name of the foundational CV model to load.
        - mode (str, optional): The mode of the model, 'eval' for evaluation or 'fine_tune' for fine-tuning. Default is 'eval'.
        """
        super(VisionModel, self).__init__()

        # Create a ViT configuration with default settings
        if not config:
            vit_config = ViTConfig()
        else:
            vit_config = config
        
        # Instantiate ViT model without pre-trained weights
        #self.vit = ViTModel(config=vit_config)
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        
        self.dropout = nn.Dropout(p=dropout_rate)
        
            
        # Set the model to evaluation or fine-tuning mode
        self.mode = mode
        if mode == 'eval':
            self.eval()
        elif mode == 'fine_tune':
            self.train()

    def forward(self, x):
        """
        Forward pass to obtain features from input data.

        Args:
        - x (torch.Tensor): Input data to obtain features from.

        Returns:
        torch.Tensor: Features extracted from the input data using the selected foundational CV model.
        """

        # Pass the input image to the model
        features = self.vit(x)
        features = features['pooler_output']
        
        features = self.dropout(features)

        # Return the features
        return features


# Add a classifier on top of the BERT model
class TextModel(torch.nn.Module):
    """
    A PyTorch module for loading and using BERT-based text models.

    This module allows you to load and use various BERT configurations for natural language processing tasks.

    Parameters:
    - config (dict): The configuration of the BERT model.
    - mode (str, optional): The mode of the model, 'eval' for evaluation or 'fine_tune' for fine-tuning. Default is 'eval'.

    Methods:
    - forward(input_ids, attention_mask): Forward pass to obtain features from input text.

    Example Usage:
    ```python
    config = {"attention_probs_dropout_prob": 0.1,
              "hidden_dropout_prob": 0.1,
              "hidden_size": 768,
              "initializer_range": 0.02,
              "intermediate_size": 3072,
              "layer_norm_eps": 1e-12,
              "max_position_embeddings": 512,
              "model_type": "bert",
              "num_attention_heads": 12,
              "num_hidden_layers": 12,
              "pad_token_id": 0,
              "type_vocab_size": 2,
              "vocab_size": 30522}
    text_model = TextModel(config)
    features = text_model(input_ids, attention_mask)
    ```

    Note:
    - This module provides access to a BERT architecture for text processing.
    - It allows for both evaluation and fine-tuning modes.

    Dependencies:
    - PyTorch
    - Hugging Face Transformers (for BERT)

    For more information on specific models, refer to the respective model's documentation.
    """
    def __init__(self, config=None, mode='fine_tune', dropout_rate=0.2):
        super(TextModel, self).__init__()
        
        # Create a ViT configuration with default settings
        if not config:
            bert_config = BertConfig()
        else:
            bert_config = config
        
        # Instantiate BERT model without pre-trained weights
        #self.bert = BertModel(config=bert_config)
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        
        self.dropout = nn.Dropout(p=dropout_rate)
            
        # Set the model to evaluation or fine-tuning mode
        self.mode = mode
        if mode == 'eval':
            self.eval()
        elif mode == 'fine_tune':
            self.train()

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        features = outputs.last_hidden_state[:, 0, :]
        features = self.dropout(features)
        return features


# Early Fusion Model
class EarlyFusionModel(nn.Module):
    """
    Early Fusion Model for combining text and image features.

    Args:
    - text_model (torch.nn.Module): Text model for extracting text features.
    - image_model (torch.nn.Module): Image model for extracting image features.
    - output_size (int): Dimension of the output.
    - hidden (int or list): Hidden layer(s) size(s) for the model.

    Attributes:
    - fc1 (nn.Sequential): First fully connected layer(s).
    - fc2 (nn.Linear): Second fully connected layer.

    Methods:
    - forward(text, image): Forward pass of the model.

    Example:
    ```python
    text_model = TextModel(config)
    image_model = VisionModel(config)
    model = EarlyFusionModel(text_model, image_model, output_size=10, hidden=[128, 64])
    ```

    Note:
    - Make sure the dimensions of the text and image features match when combining them.

    """
    def __init__(self, text_model, image_model, output_size, hidden=[128], freeze_backbone=True, p=0.2):
        super(EarlyFusionModel, self).__init__()
        
        self.p = p
        output_dim = 768 + 768
        
        self.text_model = text_model
        self.text_model.train()
        self.image_model = image_model
        self.image_model.train()
        
        if freeze_backbone:
            # Freeze the parameters of the text model
            for param in text_model.parameters():
                param.requires_grad = False

            # Freeze the parameters of the image model
            for param in image_model.parameters():
                param.requires_grad = False
        
        # Initialize layers as an empty list
        layers = []
        
        # Add the linear layer and ReLU activation if 'hidden' is an integer
        if isinstance(hidden, int):
            layers.append(nn.Linear(output_dim, hidden))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=self.p))

            output_dim = hidden
            
        # Add the linear layer and ReLU activation for each element in 'hidden' if it's a list
        elif isinstance(hidden, list):
            for h in hidden:
                layers.append(nn.Linear(output_dim, h))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(p=self.p))
                layers.append(nn.BatchNorm1d(h))
                output_dim = h
        
        self.fc1 = nn.Sequential(*layers)
        
        #self.fc1 = nn.Linear(text_input_size + image_input_size, hidden)
        
        self.fc2 = nn.Linear(output_dim, output_size)

    def forward(self, input_ids, attention_mask, image):
        """
        Forward pass of the model.

        Args:
        - text (dict): Input dictionary for text features.
        - image (torch.Tensor): Input tensor for image features.

        Returns:
        - torch.Tensor: Output of the model.
        """
        
        text_embed = self.text_model(input_ids, attention_mask)
        image_embed = self.image_model(image)
        
        x = torch.cat((text_embed, image_embed), dim=1)
        #x = torch.relu(self.fc1(x))
        x = self.fc1(x)
        x = self.fc2(x)
        return x

# Late Fusion Model
class LateFusionModel(nn.Module):
    """
    Late Fusion Model for combining text and image features.

    Args:
    - text_model (torch.nn.Module): Text model for extracting text features.
    - image_model (torch.nn.Module): Image model for extracting image features.
    - output_size (int): Dimension of the output.
    - hidden_images (int or list): Hidden layer(s) size(s) for the image features.
    - hidden_text (int or list): Hidden layer(s) size(s) for the text features.

    Attributes:
    - text_fc (nn.Sequential): Fully connected layers for text features.
    - image_fc (nn.Sequential): Fully connected layers for image features.
    - fc2 (nn.Linear): Second fully connected layer.

    Methods:
    - forward(text, image): Forward pass of the model.

    Example:
    ```python
    text_model = TextModel(config)
    image_model = VisionModel(config)
    model = LateFusionModel(text_model, image_model, output_size=10, hidden_images=[64], hidden_text=[64])
    ```

    Note:
    - Make sure the dimensions of the text and image features match when combining them.

    """
    def __init__(self, text_model, image_model, output_size, hidden_images=[64], hidden_text=[64], freeze_backbone=True, p=0.2):
        super(LateFusionModel, self).__init__()
        
        
        self.p = p
        embed_dim = 768
        
        self.text_model = text_model
        self.text_model.train()
        self.image_model = image_model
        self.image_model.train()
        
        if freeze_backbone:
            # Freeze the parameters of the text model
            for param in text_model.parameters():
                param.requires_grad = False

            # Freeze the parameters of the image model
            for param in image_model.parameters():
                param.requires_grad = False
        
        self.text_fc, out_text = self._get_layers(embed_dim, hidden_text, p=self.p)
        self.image_fc, out_images = self._get_layers(embed_dim, hidden_images, p=self.p)
        
        #self.text_fc = nn.Linear(text_input_size, hidden_text)
        #self.image_fc = nn.Linear(image_input_size, hidden_images)
        
        
        self.fc2 = nn.Linear(out_text + out_images, output_size)
        
    def _get_layers(self, embed_dim, hidden, p=0.2):
        # Initialize layers as an empty list
        layers = []
        output_dim = embed_dim
        
        # Add the linear layer and ReLU activation if 'hidden' is an integer
        if isinstance(hidden, int):
            layers.append(nn.Linear(output_dim, hidden))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=p))

            output_dim = hidden
            
        # Add the linear layer and ReLU activation for each element in 'hidden' if it's a list
        elif isinstance(hidden, list):
            for h in hidden:
                layers.append(nn.Linear(output_dim, h))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(p=p))
                layers.append(nn.BatchNorm1d(h))
                output_dim = h
        
        fc = nn.Sequential(*layers)
        
        return fc, output_dim

    def forward(self, input_ids, attention_mask, image):
        """
        Forward pass of the model.

        Args:
        - text (dict): Input dictionary for text features.
        - image (torch.Tensor): Input tensor for image features.

        Returns:
        - torch.Tensor: Output of the model.
        """
        text_embed = self.text_model(input_ids, attention_mask)
        image_embed = self.image_model(image)
        
        text_output = self.text_fc(text_embed)
        image_output = self.image_fc(image_embed)
        #text_output = torch.relu(self.text_fc(text))
        #image_output = torch.relu(self.image_fc(image))
        x = torch.cat((text_output, image_output), dim=1)
        x = self.fc2(x)
        return x


    
def test_model(y_test, y_pred):
    """
    Evaluates the model on the training and test data respectively
    1. Predictions on test data
    2. Classification report
    3. Confusion matrix
    4. ROC curve

    Inputs:
    y_test: numpy array with test labels
    y_pred: numpy array with predicted test labels
    """
    
    plot_matrix = False
    if y_pred.shape[1] < 102:
        plot_matrix = True
        
    if y_pred.shape[1] > 1:
        y_test = np.argmax(y_test, axis=1)
        y_pred = np.argmax(y_pred, axis=1)
    
    # Confusion matrix
    if plot_matrix:
        # Create a confusion matrix of the test predictions
        cm = confusion_matrix(y_test, y_pred)
        # create heatmap
        # Set the size of the plot
        fig, ax = plt.subplots(figsize=(15, 15))
        sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', ax=ax)
        # Set plot labels
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        # Display plot
        plt.show()

    #create ROC curve
    from sklearn.preprocessing import LabelBinarizer
    fig, ax = plt.subplots(figsize=(15, 15))

    label_binarizer = LabelBinarizer().fit(y_test)
    y_onehot_test = label_binarizer.transform(y_test)
    y_onehot_pred = label_binarizer.transform(y_pred)
    
    if (y_onehot_pred.shape[1] < 2):
        fpr, tpr, _ = roc_curve(y_test,  y_pred)

        #create ROC curve
        #plt.plot(fpr,tpr)
        RocCurveDisplay.from_predictions(
                y_test,
                y_pred,
                name=f"ROC curve",
                color='aqua',
                ax=ax,
            )
        plt.plot([0, 1], [0, 1], "k--", label="ROC curve for chance level (AUC = 0.5)")
        plt.title('ROC Curve')
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()
        
    else:
        from itertools import cycle
        colors = cycle(["aqua", "darkorange", "cornflowerblue", "red", "green", "yellow", "purple", "pink", "brown", "black"])

        for class_id, color in zip(range(len(label_binarizer.classes_)), colors):
            RocCurveDisplay.from_predictions(
                y_onehot_test[:, class_id],
                y_onehot_pred[:, class_id],
                name=f"ROC curve for {label_binarizer.classes_[class_id]}",
                color=color,
                ax=ax,
            )

        plt.plot([0, 1], [0, 1], "k--", label="ROC curve for chance level (AUC = 0.5)")
        plt.axis("square")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Extension of Receiver Operating Characteristic\nto One-vs-Rest multiclass")
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1), bbox_transform=plt.gcf().transFigure)
        plt.show()
        
    # Classification report
    # Create a classification report of the test predictions
    cr = classification_report(y_test, y_pred)
    # print classification report
    print(cr)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')  # Use weighted average for multi-class precision
    recall = recall_score(y_test, y_pred, average='weighted')  # Use weighted average for multi-class recall
    f1 = f1_score(y_test, y_pred, average='weighted')  # Use weighted average for multi-class F1-score

    return accuracy, precision, recall, f1

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
def train_early_fusion(train_loader, test_loader, output_size, num_epochs=5, multilabel=True, report=False, lr=0.001, adam=False, set_weights=True, freeze_backbone=True, p=0.0, device="cuda"):
    """
    Train an Early Fusion Model.

    Args:
    - train_loader (DataLoader): DataLoader for the training set.
    - test_loader (DataLoader): DataLoader for the testing set.
    - text_input_size (int): Dimension of the text input.
    - image_input_size (int): Dimension of the image input.
    - output_size (int): Dimension of the output.
    - num_epochs (int): Number of training epochs.
    - multilabel (bool): Flag for multilabel classification.
    - report (bool): Flag to generate a classification report, confusion matrix, and ROC curve.

    Example:
    train_early_fusion(train_loader, test_loader, text_input_size=512, image_input_size=256, output_size=10, num_epochs=5, multilabel=True)
    """
    
    if device not in ['cuda', 'cpu']:
        raise ValueError("Invalid device name. Please provide 'cuda' or 'cpu'.")

    text_model = TextModel()
    image_model = VisionModel()
    model = EarlyFusionModel(text_model=text_model, image_model=image_model, output_size=output_size, freeze_backbone=freeze_backbone, p=p)
    # Wrap the model with DataParallel
    model = nn.DataParallel(model)
    
    model.to(device)
        
    print(f'The number of parameters of the model are: {count_parameters(model)}')

    if set_weights:
        if not multilabel:
            # Assuming train_loader.dataset.labels is a one-hot representation
            class_indices = np.argmax(train_loader.dataset.labels, axis=1)

            # Compute class weights using class indices
            class_weights = compute_class_weight('balanced', classes=np.unique(class_indices), y=class_indices)
            class_weights = torch.tensor(class_weights, dtype=torch.float32)
        else:
            class_counts = train_loader.dataset.labels.sum(axis=0)
            total_samples = len(train_loader.dataset.labels)
            num_classes = train_loader.dataset.labels.shape[1]
            class_weights = total_samples / (num_classes * class_counts)

            # Convert class_weights to a PyTorch tensor
            class_weights = torch.tensor(class_weights, dtype=torch.float32)
    else:
        class_weights = None
    if multilabel:
        criterion = nn.BCEWithLogitsLoss(weight=class_weights)
    elif(output_size == 1):
        criterion = nn.BCEWithLogitsLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
    if adam:
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=lr)

    train_accuracy_list = []
    test_accuracy_list = []
    f1_accuracy_list = []
    
    model = model.to(device)
    criterion = criterion.to(device)
    
    # Initialize variables to store total training and inference times
    total_training_time = 0
    total_inference_time = 0
    
    for epoch in range(num_epochs):
        
        # Start measuring training time
        epoch_start_time = time.time()
        
        model.train()
        for batch in train_loader:
            input_ids, attention_mask, image, labels = batch['text']['input_ids'].to(device), batch['text']['attention_mask'].to(device), batch['image'].to(device), batch['labels'].to(device)
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask, image)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
        # End measuring training time
        epoch_end_time = time.time()
        epoch_training_time = epoch_end_time - epoch_start_time
        total_training_time += epoch_training_time
        
        # Start measuring inference time
        epoch_start_time = time.time()
        
        model.eval()
        with torch.no_grad():
            y_true, y_pred = [], []
            for batch in test_loader:
                input_ids, attention_mask, image, labels = batch['text']['input_ids'].to(device), batch['text']['attention_mask'].to(device), batch['image'].to(device), batch['labels'].to(device)
                outputs = model(input_ids, attention_mask, image)
                if multilabel or (output_size == 1):
                    preds = torch.sigmoid(outputs)
                else:
                    preds = torch.softmax(outputs, dim=1)

                y_true.extend(labels.numpy())
                y_pred.extend(preds.numpy())

            y_true, y_pred = np.array(y_true), np.array(y_pred)
            
            if multilabel or (output_size == 1):
                y_pred_one_hot = (y_pred > 0.5).astype(int)
            else:
                predicted_class_indices = np.argmax(y_pred, axis=1)
                # Convert the predicted class indices to one-hot encoding
                y_pred_one_hot = np.eye(y_pred.shape[1])[predicted_class_indices]
            
            test_accuracy = accuracy_score(y_true, y_pred_one_hot)
            f1 = f1_score(y_true, y_pred_one_hot, average='macro')
            test_accuracy_list.append(test_accuracy)
            f1_accuracy_list.append(f1)

            print(f"Epoch {epoch + 1}/{num_epochs} - Test Accuracy: {test_accuracy:.4f}, macro-f1: {f1:.4f}")
            
        # End measuring inference time
        epoch_end_time = time.time()
        epoch_inference_time = epoch_end_time - epoch_start_time
        total_inference_time += epoch_inference_time
        
        # Print or log the training and inference times for the current epoch
        print(f"Epoch {epoch + 1}/{num_epochs} - Training Time: {epoch_training_time:.2f} seconds | Inference Time: {epoch_inference_time:.2f} seconds")

    # Calculate average training time per epoch
    average_training_time_per_epoch = total_training_time / num_epochs

    # Calculate average inference time per epoch
    average_inference_time_per_epoch = total_inference_time / num_epochs

    print(f"Average Training Time per Epoch: {average_training_time_per_epoch:.2f} seconds")
    print(f"Total Training Time per Epoch: {total_training_time:.2f} seconds")
    print(f"Average Inference Time per Epoch: {average_inference_time_per_epoch:.2f} seconds")
    print(f"Total Inference Time per Epoch: {total_inference_time:.2f} seconds")
    

    # Plot the accuracy
    plt.plot(range(1, num_epochs + 1), f1_accuracy_list, label='Test F1')
    plt.plot(range(1, num_epochs + 1), test_accuracy_list, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    
    if report:
        # Evaluate the model using the test_model function after training
        model.eval()
        with torch.no_grad():
            y_true, y_pred = [], []
            for batch in test_loader:
                input_ids, attention_mask, image, labels = batch['text']['input_ids'].to(device), batch['text']['attention_mask'].to(device), batch['image'].to(device), batch['labels'].to(device)
                outputs = model(input_ids, attention_mask, image)
                if multilabel or (output_size == 1):
                    preds = torch.sigmoid(outputs)
                else:
                    preds = torch.softmax(outputs, dim=1)
                y_true.extend(labels.numpy())
                y_pred.extend(preds.numpy())

            y_true, y_pred = np.array(y_true), np.array(y_pred)
            if multilabel or (output_size == 1):
                y_pred_one_hot = (y_pred > 0.5).astype(int)
            else:
                predicted_class_indices = np.argmax(y_pred, axis=1)
                # Convert the predicted class indices to one-hot encoding
                y_pred_one_hot = np.eye(y_pred.shape[1])[predicted_class_indices]
                #test_model(y_true, (y_pred > 0.5).astype(int))
            test_model(y_true, y_pred_one_hot)
            

# Function to train late fusion model (similar changes)
def train_late_fusion(train_loader, test_loader, output_size, num_epochs=5, multilabel=True, report=False, lr=0.001, adam=False, set_weights=True, freeze_backbone=True, p=0.0, device = "cuda"):
    """
    Train a Late Fusion Model.

    Args:
    - train_loader (DataLoader): DataLoader for the training set.
    - test_loader (DataLoader): DataLoader for the testing set.
    - text_input_size (int): Dimension of the text input.
    - image_input_size (int): Dimension of the image input.
    - output_size (int): Dimension of the output.
    - num_epochs (int): Number of training epochs.
    - multilabel (bool): Flag for multilabel classification.
    - report (bool): Flag to generate a classification report, confusion matrix, and ROC curve.

    Example:
    train_late_fusion(train_loader, test_loader, text_input_size=512, image_input_size=256, output_size=10, num_epochs=5, multilabel=True)
    """
    if device not in ['cuda', 'cpu']:
        raise ValueError("Invalid device name. Please provide 'cuda' or 'cpu'.")

    # Set the device
    device = torch.device(device)
    text_model = TextModel()
    image_model = VisionModel()
    model = LateFusionModel(text_model=text_model, image_model=image_model, output_size=output_size, freeze_backbone=freeze_backbone, p=p)
    # Wrap the model with DataParallel
    model = nn.DataParallel(model)
    
    model.to(device)
    
    print(f'The number of parameters of the model are: {count_parameters(model)}')
    
    
    if set_weights:
        if not multilabel:
            # Assuming train_loader.dataset.labels is a one-hot representation
            class_indices = np.argmax(train_loader.dataset.labels, axis=1)

            # Compute class weights using class indices
            class_weights = compute_class_weight('balanced', classes=np.unique(class_indices), y=class_indices)
            class_weights = torch.tensor(class_weights, dtype=torch.float32)
        else:
            class_counts = train_loader.dataset.labels.sum(axis=0)
            total_samples = len(train_loader.dataset.labels)
            num_classes = train_loader.dataset.labels.shape[1]
            class_weights = total_samples / (num_classes * class_counts)

            # Convert class_weights to a PyTorch tensor
            class_weights = torch.tensor(class_weights, dtype=torch.float32)
    else:
        class_weights = None
        
        
    if multilabel:
        criterion = nn.BCEWithLogitsLoss(weight=class_weights)
    elif(output_size == 1):
        criterion = nn.BCEWithLogitsLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        
    if adam:
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=lr)

        
    train_accuracy_list = []
    test_accuracy_list = []
    f1_accuracy_list = []
    
    model = model.to(device)
    criterion = criterion.to(device)
    
    # Initialize variables to store total training and inference times
    total_training_time = 0
    total_inference_time = 0

    for epoch in range(num_epochs):
                
        # Start measuring training time
        epoch_start_time = time.time()
        
        model.train()
        for batch in train_loader:
            input_ids, attention_mask, image, labels = batch['text']['input_ids'].to(device), batch['text']['attention_mask'].to(device), batch['image'].to(device), batch['labels'].to(device)
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask, image)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
                                    
        # End measuring training time
        epoch_end_time = time.time()
        epoch_training_time = epoch_end_time - epoch_start_time
        total_training_time += epoch_training_time
        
        # Start measuring inference time
        epoch_start_time = time.time()

        model.eval()
        with torch.no_grad():
            y_true, y_pred = [], []
            for batch in test_loader:
                input_ids, attention_mask, image, labels = batch['text']['input_ids'].to(device), batch['text']['attention_mask'].to(device), batch['image'].to(device), batch['labels'].to(device)
                outputs = model(input_ids, attention_mask, image)
                if multilabel or (output_size == 1):
                    preds = torch.sigmoid(outputs)
                else:
                    preds = torch.softmax(outputs, dim=1)
                y_true.extend(labels.numpy())
                y_pred.extend(preds.numpy())

            y_true, y_pred = np.array(y_true), np.array(y_pred)
            if multilabel or (output_size == 1):
                y_pred_one_hot = (y_pred > 0.5).astype(int)
            else:
                predicted_class_indices = np.argmax(y_pred, axis=1)
                # Convert the predicted class indices to one-hot encoding
                y_pred_one_hot = np.eye(y_pred.shape[1])[predicted_class_indices]
            
            test_accuracy = accuracy_score(y_true, y_pred_one_hot)
            f1 = f1_score(y_true, y_pred_one_hot, average='macro')
            test_accuracy_list.append(test_accuracy)
            f1_accuracy_list.append(f1)

            print(f"Epoch {epoch + 1}/{num_epochs} - Test Accuracy: {test_accuracy:.4f}, macro-f1: {f1:.4f}")
            
        # End measuring inference time
        epoch_end_time = time.time()
        epoch_inference_time = epoch_end_time - epoch_start_time
        total_inference_time += epoch_inference_time
        
        # Print or log the training and inference times for the current epoch
        print(f"Epoch {epoch + 1}/{num_epochs} - Training Time: {epoch_training_time:.2f} seconds | Inference Time: {epoch_inference_time:.2f} seconds")

    # Calculate average training time per epoch
    average_training_time_per_epoch = total_training_time / num_epochs

    # Calculate average inference time per epoch
    average_inference_time_per_epoch = total_inference_time / num_epochs

    print(f"Average Training Time per Epoch: {average_training_time_per_epoch:.2f} seconds")
    print(f"Total Training Time per Epoch: {total_training_time:.2f} seconds")
    print(f"Average Inference Time per Epoch: {average_inference_time_per_epoch:.2f} seconds")
    print(f"Total Inference Time per Epoch: {total_inference_time:.2f} seconds")
    


    # Plot the accuracy
    plt.plot(range(1, num_epochs + 1), f1_accuracy_list, label='Test F1')
    plt.plot(range(1, num_epochs + 1), test_accuracy_list, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    
    if report:
        # Evaluate the model using the test_model function after training
        model.eval()
        with torch.no_grad():
            y_true, y_pred = [], []
            for batch in test_loader:
                input_ids, attention_mask, image, labels = batch['text']['input_ids'].to(device), batch['text']['attention_mask'].to(device), batch['image'].to(device), batch['labels'].to(device)
                outputs = model(input_ids, attention_mask, image)
                if multilabel or (output_size == 1):
                    preds = torch.sigmoid(outputs)
                else:
                    preds = torch.softmax(outputs, dim=1)
                y_true.extend(labels.numpy())
                y_pred.extend(preds.numpy())
                
            y_true, y_pred = np.array(y_true), np.array(y_pred)
            
            if multilabel or (output_size == 1):
                y_pred_one_hot = (y_pred > 0.5).astype(int)
            else:
                predicted_class_indices = np.argmax(y_pred, axis=1)
                # Convert the predicted class indices to one-hot encoding
                y_pred_one_hot = np.eye(y_pred.shape[1])[predicted_class_indices]
                #test_model(y_true, (y_pred > 0.5).astype(int))
            
            test_model(y_true, y_pred_one_hot)
