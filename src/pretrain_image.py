#!/usr/bin/env python
import os
import argparse
import subprocess
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
from .cv_data_loader import ImageFolderDataset
from .cv_models import FoundationalCVModel
from tqdm import tqdm

####################
# Custom Dataset   #
####################
class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None, image_files=None):
        self.data_dir = data_dir
        self.folder_path = data_dir
        self.transform = transform
                
        if image_files:
            self.image_files = image_files
            self.clean_unidentified_images()
        else:
            self.image_files = [f for f in os.listdir(self.folder_path) if f.lower().endswith(('jpg', 'jpeg', 'png', 'gif'))]

    def clean_unidentified_images(self):
        """
        Clean the dataset by removing images causing UnidentifiedImageError.
        """
        cleaned_files = []
        for img_name in self.image_files:
            img_path = os.path.join(self.folder_path, img_name)
            try:
                Image.open(img_path).convert("RGB")
                cleaned_files.append(img_name)
            except:
                Image.open(img_path).convert("RGB")
                print(f"Skipping {img_name} due to error")
        self.image_files = cleaned_files
    
                
    def __len__(self):
        return len(self.image_files)
        
    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image


#####################################
# Data Augmentation (Two Views)     #
#####################################
def get_two_views(image):
    """
    Returns two differently augmented versions of the input PIL image.
    """
    transform1 = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.ToTensor()
    ])
    transform2 = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.ToTensor()
    ])
    return transform1(image), transform2(image)


def get_two_views(image):
    """
    Returns two differently augmented versions of the input PIL image.
    Mimics DINO augmentations with bicubic interpolation, color jitter,
    Gaussian blur, and solarization.
    """
    normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    # Global view 1
    view1 = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.4, 1.0), interpolation=Image.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                   saturation=0.2, hue=0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=23),
        normalize,
    ])(image)
    # Global view 2 with solarization
    view2 = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.4, 1.0), interpolation=Image.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                   saturation=0.2, hue=0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=23),
        transforms.RandomSolarize(threshold=128, p=0.2),
        normalize,
    ])(image)
    return view1, view2



##############################
# SSL Models and Losses      #
##############################

# --- Contrastive Learning (SimCLR-style) ---
class ContrastiveModel(nn.Module):
    def __init__(self, backbone):
        super(ContrastiveModel, self).__init__()
        self.backbone = backbone

    def forward(self, x1, x2):
        z1 = self.backbone(x1)
        z2 = self.backbone(x2)
        return z1, z2

def contrastive_loss(z1, z2, temperature=0.5):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    logits = torch.matmul(z1, z2.T) / temperature
    labels = torch.arange(z1.size(0)).to(z1.device)
    loss = F.cross_entropy(logits, labels)
    return loss


# MLP to serve as both the projector and predictor.
class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)


def get_backbone_output_dim(backbone, input_size=(3, 224, 224), device='cuda'):
    # Create a dummy input tensor
    dummy_input = torch.randn(1, *input_size).to(device)
    
    # Set the backbone to evaluation mode and disable gradients
    backbone.eval()
    with torch.no_grad():
        features = backbone(dummy_input)
    
    # If the output is a multi-dimensional tensor (e.g., with spatial dimensions),
    # you might want to flatten it (excluding the batch dimension)
    if features.dim() > 2:
        features = torch.flatten(features, 1)
    
    # Return the feature dimension (the last dimension)
    return features.shape[-1]


# --- BYOL ---
class BYOL(nn.Module):
    def __init__(self, backbone, projection_dim=256, hidden_dim=4096, in_dim=None, device='cuda', input_size=(3, 224, 224)):
        super(BYOL, self).__init__()
        # Online network components
        backbone = backbone.to(device)
        self.online_encoder = backbone
        
        # If not in_dim is provided, use the default output dimension of the backbone
        in_dim = get_backbone_output_dim(backbone, input_size=input_size, device=device) if in_dim is None else in_dim
        
        self.online_projector = MLP(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=projection_dim)
        self.online_predictor = MLP(in_dim=projection_dim, hidden_dim=hidden_dim // 4, out_dim=projection_dim)
        
        # Target network components (encoder and projector only)
        self.target_encoder = copy.deepcopy(backbone)
        self.target_projector = copy.deepcopy(self.online_projector)
        # Freeze target network parameters
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        for param in self.target_projector.parameters():
            param.requires_grad = False

    def forward(self, x1, x2):
        # Online network forward pass
        online_proj_1 = self.online_projector(self.online_encoder(x1))
        online_proj_2 = self.online_projector(self.online_encoder(x2))
        online_pred_1 = self.online_predictor(online_proj_1)
        online_pred_2 = self.online_predictor(online_proj_2)
        
        # Target network forward pass (without gradient)
        with torch.no_grad():
            target_proj_1 = self.target_projector(self.target_encoder(x1))
            target_proj_2 = self.target_projector(self.target_encoder(x2))
        
        return online_pred_1, online_pred_2, target_proj_1, target_proj_2



# BYOL loss function (cosine similarity loss)
def byol_loss(pred, target):
    pred = F.normalize(pred, dim=-1)
    target = F.normalize(target, dim=-1)
    return 2 - 2 * (pred * target).sum(dim=-1).mean()

# Momentum update for the target network
def update_momentum(model, target_model, momentum=0.99):
    for param, target_param in zip(model.parameters(), target_model.parameters()):
        target_param.data = momentum * target_param.data + (1.0 - momentum) * param.data



# --- DINO ---
# MLP head for DINO (3-layer MLP with weight normalization and output normalization)
class MLP_DINO(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(MLP_DINO, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )
        # Apply weight normalization to the last layer
        self.net[-1] = nn.utils.weight_norm(self.net[-1])
    
    def forward(self, x):
        x = self.net(x)
        # L2-normalize the output along the feature dimension
        return F.normalize(x, dim=-1)

# Updated DINO class integrating teacher/student heads and a center buffer
class DINO(nn.Module):
    def __init__(self, backbone, out_dim=256, hidden_dim=2048, 
                 student_temp=0.1, teacher_temp=0.07, center_momentum=0.9,
                 in_dim=None, device='cuda', input_size=(3, 224, 224)):
        super(DINO, self).__init__()
        
        backbone = backbone.to(device)
        self.student = backbone
        self.teacher = copy.deepcopy(backbone)
        for param in self.teacher.parameters():
            param.requires_grad = False
            
        # If not in_dim is provided, use the default output dimension of the backbone
        in_dim = get_backbone_output_dim(backbone, input_size=input_size, device=device) if in_dim is None else in_dim
        #in_dim = get_backbone_output_dim(backbone)
        
        self.student_head = MLP_DINO(in_dim, hidden_dim, out_dim)
        self.teacher_head = MLP_DINO(in_dim, hidden_dim, out_dim)
        
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.center_momentum = center_momentum
        # Center buffer to be subtracted from teacher outputs before temperature scaling
        self.register_buffer("center", torch.zeros(1, out_dim))
    
    def forward(self, x1, x2):
        # Compute student predictions for both views
        s1 = self.student_head(self.student(x1))
        s2 = self.student_head(self.student(x2))
        # Compute teacher outputs without gradient (for stability)
        with torch.no_grad():
            t1 = self.teacher_head(self.teacher(x1))
            t2 = self.teacher_head(self.teacher(x2))
        return s1, s2, t1, t2

# Updated DINO loss function using cross-entropy loss between teacher and student predictions
def dino_loss(student, teacher, student_temp, teacher_temp, center):
    # Scale student logits with student temperature
    student_logits = student / student_temp
    # Center and scale teacher logits with teacher temperature
    teacher_logits = (teacher - center) / teacher_temp
    # Teacher probabilities (no gradients)
    teacher_probs = teacher_logits.detach().softmax(dim=-1)
    # Student log probabilities
    student_log_probs = student_logits.log_softmax(dim=-1)
    loss = - (teacher_probs * student_log_probs).sum(dim=-1).mean()
    return loss


def update_teacher(student, teacher, momentum=0.99):
    for param_s, param_t in zip(student.parameters(), teacher.parameters()):
        param_t.data = param_t.data * momentum + param_s.data * (1.0 - momentum)

###################################
# Training Routine Implementation #
###################################
def train_ssl(method, dataloader, model, optimizer, device, epochs):
    model.to(device)
    inner = model.module if isinstance(model, nn.DataParallel) else model
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        #progress_bar = tqdm(dataloader, desc=f"Epoch [{epoch+1}/{epochs}]")
        for images in tqdm(dataloader, desc=f"Epoch [{epoch+1}/{epochs}]"):
            images = images.to(device)
            x1_batch, x2_batch = [], []
            for img in images:
                view1, view2 = get_two_views(transforms.ToPILImage()(img.cpu()))
                x1_batch.append(view1)
                x2_batch.append(view2)
            x1 = torch.stack(x1_batch).to(device)
            x2 = torch.stack(x2_batch).to(device)
            optimizer.zero_grad()
            if method == 'contrastive':
                z1, z2 = model(x1, x2)
                loss = contrastive_loss(z1, z2)
            elif method == 'byol':
                online_pred_1, online_pred_2, target_proj_1, target_proj_2 = model(x1, x2)
                loss = byol_loss(online_pred_1, target_proj_2) + byol_loss(online_pred_2, target_proj_1)
            
            elif method == 'dino':
                s1, s2, t1, t2 = model(x1, x2)
                # use inner.student_temp etc. instead of model.student_temp
                loss = dino_loss(s1, t2, inner.student_temp, inner.teacher_temp, inner.center) + \
                     dino_loss(s2, t1, inner.student_temp, inner.teacher_temp, inner.center)
                #loss = dino_loss(s1, t2, model.student_temp, model.teacher_temp, model.center) + \
                #    dino_loss(s2, t1, model.student_temp, model.teacher_temp, model.center)

            else:
                raise ValueError("Unsupported method")
            loss.backward()
            optimizer.step()
            if method == 'byol':
                update_momentum(inner.online_encoder, inner.target_encoder, momentum=0.99)
                update_momentum(inner.online_projector, inner.target_projector, momentum=0.99)
                #update_momentum(model.online_encoder, model.target_encoder, momentum=0.99)
                #update_momentum(model.online_projector, model.target_projector, momentum=0.99)
            elif method == 'dino':
                # Update teacher network parameters with EMA
                update_teacher(inner.student, inner.teacher, momentum=0.996)
                #update_teacher(model.student, model.teacher, momentum=0.996)
                # Update the center buffer using an exponential moving average
                with torch.no_grad():
                    batch_center = torch.cat([t1, t2], dim=0).mean(dim=0, keepdim=True)
                    inner.center = inner.center * inner.center_momentum + batch_center * (1 - inner.center_momentum)
                    
                    #model.center = model.center * model.center_momentum + batch_center * (1 - model.center_momentum)
                    
            total_loss += loss.item()
            
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader):.4f}")
    return model

####################################
# Main pretrain_model Function     #
####################################
def pretrain_model(batch_size, path, dataset_name, backbone, directory, device="cuda", method="byol", epochs=10, image_files=None):
    """
    Pretrains a foundational CV model using SSL on a given image dataset.
    
    Parameters:
      - batch_size: Batch size for training.
      - path: Path to the image dataset.
      - dataset_name: Name of the dataset (for logging purposes).
      - backbone: Backbone model name (e.g., 'vit_base', 'dinov2_base', etc.).
      - directory: File path where the pretrained weights will be saved.
      - device: 'cuda' or 'cpu'.
      - method: SSL method to use ('contrastive', 'byol', or 'dino').
      - epochs: Number of training epochs.
      
    Returns:
      The path to the saved weights.
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(device)
    # Simple transform; additional augmentations are applied in get_two_views
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    dataset = CustomDataset(data_dir=path, transform=transform, image_files=image_files)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    # Initialize the foundational model
    backbone_model = FoundationalCVModel(backbone=backbone, mode='fine_tune')
    
    # Wrap the backbone with the chosen SSL method
    if method == 'contrastive':
        model = ContrastiveModel(backbone_model)
    elif method == 'byol':
        model = BYOL(backbone_model)
    elif method == 'dino':
        model = DINO(backbone_model)
    else:
        raise ValueError("Unsupported method")
    
    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    if torch.cuda.device_count() > 1:
        print(f"Found {torch.cuda.device_count()} GPUs, using DataParallel")
        model = nn.DataParallel(model)
    
    model.to(device)
    
    print(f"Starting SSL training on dataset {dataset_name} using method: {method} with backbone: {backbone}")
    model = train_ssl(method, dataloader, model, optimizer, device, epochs)

    # Save the backbone weights (for DINO, we save the student network)
    if method in ['contrastive', 'byol']:
        torch.save(backbone_model.state_dict(), directory)
    elif method == 'dino':
        # if wrapped in DataParallel:
        if isinstance(model, nn.DataParallel):
            to_save = model.module.student
        else:
            to_save = model.student
 
        torch.save(to_save.state_dict(), directory)
        
    print(f"Backbone weights saved to {directory}")
    return directory

