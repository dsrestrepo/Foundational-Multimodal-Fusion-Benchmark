from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from mpl_toolkits.mplot3d import Axes3D

import os
import numpy as np
import pandas as pd


def normalize_embeddings(embeddings):
    """Normalize embeddings to the unit sphere."""
    return normalize(embeddings, axis=1, norm='l2')

def modify_and_normalize_embeddings(text_embeddings, image_embeddings, lambda_shift):
    """Shift and normalize embeddings."""
    # Calculate the original gap vector
    gap_vector = np.mean(image_embeddings, axis=0) - np.mean(text_embeddings, axis=0)
    
    # Shift embeddings
    text_embeddings_shifted = text_embeddings + (lambda_shift/2) * gap_vector
    image_embeddings_shifted = image_embeddings - (lambda_shift/2) * gap_vector
    
    # Normalize to the unit sphere
    text_embeddings_shifted = normalize_embeddings(text_embeddings_shifted)
    image_embeddings_shifted = normalize_embeddings(image_embeddings_shifted)
    
    return text_embeddings_shifted, image_embeddings_shifted


def visualize_embeddings(text_embeddings, image_embeddings, title, lambda_shift, DATASET):
    """Visualize embeddings in 2D and 3D, including the unit circle and sphere."""
    pca = PCA(n_components=2)
    all_embeddings = np.concatenate([text_embeddings, image_embeddings])
    reduced_embeddings = pca.fit_transform(all_embeddings)
    
    # Split reduced embeddings back
    reduced_text_embeddings = reduced_embeddings[:len(text_embeddings)]
    reduced_image_embeddings = reduced_embeddings[len(text_embeddings):]
    
    # Plotting in 2D with unit circle
    plt.figure(figsize=(10, 6))
    circle = plt.Circle((0, 0), 1, color='green', fill=False)
    plt.gca().add_artist(circle)
    plt.scatter(reduced_text_embeddings[:, 0], reduced_text_embeddings[:, 1], label='Text Embeddings', alpha=0.5)
    plt.scatter(reduced_image_embeddings[:, 0], reduced_image_embeddings[:, 1], label='Image Embeddings', alpha=0.5)
    plt.legend(loc='upper right')
    plt.title(title + ' in 2D')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    plt.gca().set_aspect('equal', adjustable='box')
    
    img_path_2d = f'Images/{DATASET}/2d_shift({lambda_shift}).pdf'
    os.makedirs(os.path.dirname(img_path_2d), exist_ok=True)
    plt.savefig(img_path_2d)
    plt.show()

    # Plotting in 3D with unit sphere
    fig = plt.figure(figsize=(10, 10))  # Corrected figsize
    ax = fig.add_subplot(111, projection='3d')
    pca_3d = PCA(n_components=3)
    reduced_embeddings_3d = pca_3d.fit_transform(all_embeddings)
    reduced_text_embeddings_3d = reduced_embeddings_3d[:len(text_embeddings)]
    reduced_image_embeddings_3d = reduced_embeddings_3d[len(text_embeddings):]
    
    ax.scatter(reduced_text_embeddings_3d[:, 0], reduced_text_embeddings_3d[:, 1], reduced_text_embeddings_3d[:, 2], label='Text Embeddings', alpha=0.5)
    ax.scatter(reduced_image_embeddings_3d[:, 0], reduced_image_embeddings_3d[:, 1], reduced_image_embeddings_3d[:, 2], label='Image Embeddings', alpha=0.5)
    
    # Draw a unit sphere
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = np.cos(u)*np.sin(v)
    y = np.sin(u)*np.sin(v)
    z = np.cos(v)
    ax.plot_wireframe(x, y, z, color="r", alpha=0.1)
    
    ax.set_title(title + ' in 3D')
    ax.set_xlabel('PCA Component 1', labelpad=10)
    ax.set_ylabel('PCA Component 2', labelpad=10)
    ax.set_zlabel('PCA Component 3', labelpad=10)
    plt.legend()
    
    img_path_3d = f'Images/{DATASET}/3d_shift({lambda_shift}).pdf'
    os.makedirs(os.path.dirname(img_path_3d), exist_ok=True)
    plt.savefig(img_path_3d)
    plt.show()
