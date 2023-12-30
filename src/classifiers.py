import os

import pandas as pd
import numpy as np
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

import time



######### Merge Datasets #########

def process_embeddings(df, col_name):
    """
    Process embeddings in a DataFrame column.

    Args:
    - df (pd.DataFrame): The DataFrame containing the embeddings column.
    - col_name (str): The name of the column containing the embeddings.

    Returns:
    pd.DataFrame: The DataFrame with processed embeddings.

    Steps:
    1. Convert the values in the specified column to lists.
    2. Extract values from lists and create new columns for each element.
    3. Remove the original embeddings column.

    Example:
    df_processed = process_embeddings(df, 'embeddings')
    """
    # Step 1: Convert the values in the column to lists
    df[col_name] = df[col_name].apply(eval)

    # Step 2-4: Extract values from lists and create new columns
    embeddings_df = pd.DataFrame(df[col_name].to_list(), columns=[f"text_{i+1}" for i in range(df[col_name].str.len().max())])
    df = pd.concat([df, embeddings_df], axis=1)

    # Step 5: Remove the original "embeddings" column
    df = df.drop(columns=[col_name])

    return df

def rename_image_embeddings(df):
    """
    Rename columns in a DataFrame for image embeddings.

    Args:
    - df (pd.DataFrame): The DataFrame containing columns to be renamed.

    Returns:
    pd.DataFrame: The DataFrame with renamed columns.

    Example:
    df_renamed = rename_image_embeddings(df)
    """
    df.columns = [f'image_{int(col)}' if col.isdigit() else col for col in df.columns]

    return df

# Preprocess and merge the dataframes
def preprocess_data(text_data, image_data, text_id="image_id", image_id="ImageName", embeddings_col = 'embeddings'):
    """
    Preprocess and merge text and image dataframes.

    Args:
    - text_data (pd.DataFrame): DataFrame containing text data.
    - image_data (pd.DataFrame): DataFrame containing image data.
    - text_id (str): Column name for text data identifier.
    - image_id (str): Column name for image data identifier.
    - embeddings_col (str): Column name for embeddings data.

    Returns:
    pd.DataFrame: Merged and preprocessed DataFrame.

    Steps:
    1. Process text and image embeddings.
    2. Convert image_id and text_id values to integers.
    3. Merge dataframes using image_id.
    4. Drop unnecessary columns.

    Example:
    merged_df = preprocess_data(text_df, image_df)
    """
    text_data = process_embeddings(text_data, embeddings_col)
    image_data = rename_image_embeddings(image_data)
    
    # Remove file extension from image_id
    if text_data[text_id].dtype != int:
        text_data[text_id] = text_data[text_id].apply(lambda x: int(x.split('.')[0]) if x.split('.')[0].isdigit() else x.split('.')[0])
    if image_data[image_id].dtype != int:
        image_data[image_id] = image_data[image_id].apply(lambda x: int(x.split('.')[0]) if x.split('.')[0].isdigit() else x.split('.')[0])
    #text_data[text_id] = text_data[text_id].apply(lambda x: x.split('.')[0])
    #image_data[image_id] = image_data[image_id].apply(lambda x: x.split('.')[0])

    # Merge dataframes using image_id
    df = pd.merge(text_data, image_data, left_on=text_id, right_on=image_id)

    # Drop unnecessary columns
    df.drop([image_id, text_id], axis=1, inplace=True)

    return df

######### Datasets Preparation #########

# Function to split the data into train and test
def split_data(df):
    """
    Split a DataFrame into train and test sets based on the 'split' column.

    Args:
    - df (pd.DataFrame): The DataFrame to be split.

    Returns:
    pd.DataFrame: Train set.
    pd.DataFrame: Test set.

    Example:
    train_set, test_set = split_data(df)
    """
    train_df = df[df['split'] == 'train']
    test_df = df[df['split'] == 'test']
    
    print("Train Shape:", train_df.shape)
    print("Test Shape:", test_df.shape)
    return train_df, test_df

# Function to process text labels and one-hot encode them
def process_labels(df, col='answer', mlb=None, train_columns=None):
    """
    Process text labels and perform one-hot encoding using MultiLabelBinarizer.

    Args:
    - df (pd.DataFrame): The DataFrame containing the labels.
    - col (str): The column name containing the labels.
    - mlb (sklearn.preprocessing.MultiLabelBinarizer): The MultiLabelBinarizer object.
    - train_columns (list): List of columns from the training set.

    Returns:
    pd.DataFrame: One-hot encoded labels.
    sklearn.preprocessing.MultiLabelBinarizer: MultiLabelBinarizer object.
    list: List of columns from the training set.

    Example:
    one_hot_labels, mlb, train_columns = process_labels(df, col='answer')
    """
    if mlb is None:
        mlb = MultiLabelBinarizer()
        if df[col].dtype == int:
            label = df[col]
        else:
            labels = df[col].apply(lambda x: set(x.split(', ')))
        
        if df[col].dtype == int and (len(df[col].unique()) == 2):
            train_columns = col
            one_hot_labels = label
        else:
            one_hot_labels = pd.DataFrame(mlb.fit_transform(labels), columns=mlb.classes_)
            # Save the columns from the training set
            train_columns = one_hot_labels.columns
        
        return one_hot_labels, mlb, train_columns

    else:
        if df[col].dtype == int:
            label = df[col]
        else:
            labels = df[col].apply(lambda x: set(x.split(', ')))
        
        if df[col].dtype == int and (len(df[col].unique()) == 2):
            one_hot_labels = label
        else:
            one_hot_labels = pd.DataFrame(mlb.transform(labels), columns=train_columns)
        
        return one_hot_labels

    
# Custom Dataset class for PyTorch
class VQADataset(Dataset):
    """
    Custom PyTorch Dataset for VQA (Visual Question Answering).

    Args:
    - df (pd.DataFrame): The DataFrame containing the dataset.
    - text_cols (list): List of column names containing text data.
    - image_cols (list): List of column names containing image data.
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
    def __init__(self, df, text_cols, image_cols, label_col, mlb, train_columns):
        self.text_data = df[text_cols].values
        self.image_data = df[image_cols].values
        self.mlb = mlb
        self.train_columns = train_columns
        self.labels = process_labels(df, col=label_col, mlb=mlb, train_columns=train_columns).values
        #print(self.labels.shape)
        if len(self.labels.shape) == 1:
            self.labels = np.expand_dims(self.labels, axis=1)


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'text': torch.FloatTensor(self.text_data[idx]),
            'image': torch.FloatTensor(self.image_data[idx]),
            'labels': torch.FloatTensor(self.labels[idx])
        }


######### Models and Evaluation #########

# Early Fusion Model
class EarlyFusionModel(nn.Module):
    """
    Early Fusion Model for combining text and image features.

    Args:
    - text_input_size (int): Dimension of the text input.
    - image_input_size (int): Dimension of the image input.
    - output_size (int): Dimension of the output.
    - hidden (int or list): Hidden layer(s) size(s) for the model.

    Attributes:
    - fc1 (nn.Sequential): First fully connected layer(s).
    - fc2 (nn.Linear): Second fully connected layer.

    Methods:
    - forward(text, image): Forward pass of the model.

    Example:
    model = EarlyFusionModel(text_input_size=512, image_input_size=256, output_size=10, hidden=[128, 64])
    """
    def __init__(self, text_input_size, image_input_size, output_size, hidden=[128]):
        super(EarlyFusionModel, self).__init__()
        
        output_dim = text_input_size + image_input_size
        
        # Initialize layers as an empty list
        layers = []
        
        # Add the linear layer and ReLU activation if 'hidden' is an integer
        if isinstance(hidden, int):
            layers.append(nn.Linear(output_dim, hidden))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=0.2))

            output_dim = hidden
            
        # Add the linear layer and ReLU activation for each element in 'hidden' if it's a list
        elif isinstance(hidden, list):
            for h in hidden:
                layers.append(nn.Linear(output_dim, h))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(p=0.2))
                layers.append(nn.BatchNorm1d(h))
                output_dim = h
        
        self.fc1 = nn.Sequential(*layers)

        
        #self.fc1 = nn.Linear(text_input_size + image_input_size, hidden)
        
        self.fc2 = nn.Linear(output_dim, output_size)

    def forward(self, text, image):
        x = torch.cat((text, image), dim=1)
        #x = torch.relu(self.fc1(x))
        x = self.fc1(x)
        x = self.fc2(x)
        return x

# Late Fusion Model
class LateFusionModel(nn.Module):
    """
    Late Fusion Model for combining text and image features.

    Args:
    - text_input_size (int): Dimension of the text input.
    - image_input_size (int): Dimension of the image input.
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
    model = LateFusionModel(text_input_size=512, image_input_size=256, output_size=10, hidden_images=[64], hidden_text=[64])
    """
    def __init__(self, text_input_size, image_input_size, output_size, hidden_images=[64], hidden_text=[64]):
        super(LateFusionModel, self).__init__()
        
        self.text_fc, out_text = self._get_layers(text_input_size, hidden_text)
        self.image_fc, out_images = self._get_layers(image_input_size, hidden_images)
        
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

    def forward(self, text, image):
        text_output = self.text_fc(text)
        image_output = self.image_fc(image)
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
    # Create a confusion matrix of the test predictions
    if plot_matrix:
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
    
def train_early_fusion(train_loader, test_loader, text_input_size, image_input_size, output_size, num_epochs=5, multilabel=True, report=False, lr=0.001):
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
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EarlyFusionModel(text_input_size=text_input_size, image_input_size=image_input_size, output_size=output_size)
    model = nn.DataParallel(model)
    
    model.to(device)
    
    print(f'The number of parameters of the model are: {count_parameters(model)}')
    
    from sklearn.utils.class_weight import compute_class_weight
    import numpy as np

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

    if multilabel:
        criterion = nn.BCEWithLogitsLoss(weight=class_weights)
    elif(output_size == 1):
        criterion = nn.BCEWithLogitsLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    #if multilabel or (output_size == 1):
    #    criterion = nn.BCEWithLogitsLoss()
    #else:
    #    criterion = nn.CrossEntropyLoss()
        
    #optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_accuracy_list = []
    test_accuracy_list = []
    
    # Initialize variables to store total training and inference times
    total_training_time = 0
    total_inference_time = 0
    

    for epoch in range(num_epochs):
        
        # Start measuring training time
        epoch_start_time = time.time()
        
        model.train()
        for batch in train_loader:
            text, image, labels = batch['text'].to(device), batch['image'].to(device), batch['labels'].to(device)
            optimizer.zero_grad()
            outputs = model(text, image)
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
                text, image, labels = batch['text'].to(device), batch['image'].to(device), batch['labels'].to(device)
                outputs = model(text, image)
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
                
            test_accuracy = accuracy_score(y_true, y_pred_one_hot)
            test_accuracy_list.append(test_accuracy)

            print(f"Epoch {epoch + 1}/{num_epochs} - Test Accuracy: {test_accuracy:.4f}")
            
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
    #plt.plot(range(1, num_epochs + 1), train_accuracy_list, label='Train Accuracy')
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
                text, image, labels = batch['text'].to(device), batch['image'].to(device), batch['labels'].to(device)
                outputs = model(text, image)
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
def train_late_fusion(train_loader, test_loader, text_input_size, image_input_size, output_size, num_epochs=5, multilabel=True, report=False, lr=0.001):
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LateFusionModel(text_input_size=text_input_size, image_input_size=image_input_size, output_size=output_size)
    model = nn.DataParallel(model)
    
    model.to(device)
    
    print(f'The number of parameters of the model are: {count_parameters(model)}')
    
    from sklearn.utils.class_weight import compute_class_weight
    import numpy as np

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

    if multilabel:
        criterion = nn.BCEWithLogitsLoss(weight=class_weights)
    elif(output_size == 1):
        criterion = nn.BCEWithLogitsLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
    
    #print(f'The number of parameters of the model are: {count_parameters(model)}')
    
    #if multilabel or (output_size == 1):
    #    criterion = nn.BCEWithLogitsLoss()
    #else:
    #    criterion = nn.CrossEntropyLoss()
    
    #optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    train_accuracy_list = []
    test_accuracy_list = []
    
    # Initialize variables to store total training and inference times
    total_training_time = 0
    total_inference_time = 0

    for epoch in range(num_epochs):
                
        # Start measuring training time
        epoch_start_time = time.time()
        
        model.train()
        for batch in train_loader:
            text, image, labels = batch['text'].to(device), batch['image'].to(device), batch['labels'].to(device)
            optimizer.zero_grad()
            outputs = model(text, image)
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
                text, image, labels = batch['text'].to(device), batch['image'].to(device), batch['labels'].to(device)
                outputs = model(text, image)
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
                
            test_accuracy = accuracy_score(y_true, y_pred_one_hot)
            test_accuracy_list.append(test_accuracy)

            print(f"Epoch {epoch + 1}/{num_epochs} - Test Accuracy: {test_accuracy:.4f}")
        
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
    #plt.plot(range(1, num_epochs + 1), train_accuracy_list, label='Train Accuracy')
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
                text, image, labels = batch['text'].to(device), batch['image'].to(device), batch['labels'].to(device)
                outputs = model(text, image)
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

# Function to evaluate classic ML model
def evaluate_classic_ml_model(model_name, y_true, y_pred, train_columns):
    """
    Evaluate the performance of classic ML models.

    Args:
    - model_name (str): Name of the ML model.
    - y_true (np.ndarray): True labels.
    - y_pred (np.ndarray): Predicted labels.
    - train_columns (list): List of columns from the training set.

    Example:
    evaluate_classic_ml_model("Random Forest", y_test, rf_pred, train_columns)
    """
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='micro')

    print(f"{model_name} - Test Accuracy: {accuracy}")
    print(f"{model_name} - Test F1 Score: {f1}")
    
def train_classic_ml_models(train_data, test_data, train_labels, test_labels, train_columns):
    """
    Train and evaluate classic ML models.

    Args:
    - train_data (pd.DataFrame): DataFrame containing training data.
    - test_data (pd.DataFrame): DataFrame containing testing data.
    - train_labels (np.ndarray): Labels for the training set.
    - test_labels (np.ndarray): Labels for the testing set.
    - train_columns (list): List of columns from the training set.

    Example:
    train_classic_ml_models(train_data, test_data, train_labels, test_labels, train_columns)
    """
    # Separate features and labels
    X_train, y_train = train_data[text_columns + image_columns], train_labels
    X_test, y_test = test_data[text_columns + image_columns], test_labels

    # Random Forest
    rf_model = OneVsRestClassifier(RandomForestClassifier())
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)

    # Logistic Regression
    lr_model = OneVsRestClassifier(LogisticRegression())
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)

    # SVM
    svm_model = OneVsRestClassifier(SVC())
    svm_model.fit(X_train, y_train)
    svm_pred = svm_model.predict(X_test)

    # Evaluate models
    evaluate_classic_ml_model("Random Forest", y_test, rf_pred, train_columns)
    evaluate_classic_ml_model("Logistic Regression", y_test, lr_pred, train_columns)
    evaluate_classic_ml_model("SVM", y_test, svm_pred, train_columns)
