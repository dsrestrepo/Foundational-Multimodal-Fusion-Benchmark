import os
import requests
import tarfile
import zipfile
import shutil
import pandas as pd
import gdown
from sklearn.model_selection import train_test_split
from tqdm import tqdm as tqdm
import urllib.request
import numpy as np
from urllib.error import HTTPError
import time
import matplotlib.pyplot as plt
import subprocess
import getpass
# Data reading in Dataframe format and data preprocessing
import pandas as pd
from pandas import read_csv
from pandas import DataFrame
from pandas import concat

# Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Read the images
from skimage import io
from skimage.transform import resize
from skimage.util import crop

# Linear algebra operations
import numpy as np
from skimage.io import imread, imsave

# Machine learning models and preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

# Epiweek
from epiweeks import Week, Year

# Date
from datetime import date as convert_to_date

# OS
import os

import warnings
warnings.filterwarnings('ignore')


########### DAQUAR ###########
def get_daquar_dataset(output_dir="data/"):
    """
    Downloads and stores the DAQUAR dataset.

    Parameters:
    output_dir (str, optional): The directory where the dataset will be downloaded and uncompressed. Defaults to "data/".

    Example:
    get_daquar_dataset(output_dir="data/")

    This example would download and store the DAQUAR dataset into the "data/" directory.
    """

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Download and uncompress images
        images_url = "http://datasets.d2.mpi-inf.mpg.de/mateusz14visual-turing/nyu_depth_images.tar"
        images_tar_file_path = os.path.join(output_dir, "nyu_depth_images.tar")

        response = requests.get(images_url, stream=True)
        response.raise_for_status()

        with open(images_tar_file_path, "wb") as tar_file:
            for chunk in response.iter_content(chunk_size=8192):
                tar_file.write(chunk)

        with tarfile.open(images_tar_file_path, "r:") as tar:
            # Extract to a temporary folder
            tar.extractall(output_dir)

        # Rename the extracted folder to 'images'
        extracted_folder = os.path.join(output_dir, 'nyu_depth_images')
        new_folder = os.path.join(output_dir, 'images')
        os.rename(extracted_folder, new_folder)

        print("Images downloaded and uncompressed successfully.")

        # Download labels
        train_labels_url = "http://datasets.d2.mpi-inf.mpg.de/mateusz14visual-turing/qa.894.raw.train.txt"
        test_labels_url = "http://datasets.d2.mpi-inf.mpg.de/mateusz14visual-turing/qa.894.raw.test.txt"

        train_labels_file_path = os.path.join(output_dir, "train.txt")
        test_labels_file_path = os.path.join(output_dir, "test.txt")

        response = requests.get(train_labels_url)
        response.raise_for_status()
        with open(train_labels_file_path, "wb") as train_file:
            train_file.write(response.content)

        response = requests.get(test_labels_url)
        response.raise_for_status()
        with open(test_labels_file_path, "wb") as test_file:
            test_file.write(response.content)

        print("Labels downloaded successfully.")

    except requests.RequestException as e:
        print(f"Error: {e}")
    except tarfile.TarError as e:
        print(f"Error uncompressing the dataset: {e}")
    finally:
        # Remove the downloaded tar file for images
        os.remove(images_tar_file_path)

def preprocess_daquar_dataset(output_dir="data/"):
    """
    Preprocesses the DaQuar dataset, consisting of question-answer pairs and image IDs,
    and saves the resulting DataFrame to a CSV file.

    Parameters:
    - output_dir (str): The directory where the preprocessed data will be saved. Default is "data/".

    Returns:
    None

    The function performs the following steps:
    1. Reads the training and testing data from text files in the specified directory.
    2. Extracts questions and answers from the data, creating separate DataFrames for training and testing sets.
    3. Extracts image IDs from the questions and adds them to the corresponding DataFrames.
    4. Adds a 'split' column to distinguish between training and testing data.
    5. Combines the training and testing DataFrames into a single DataFrame.
    6. Reorders the columns to have 'question', 'image_id', 'answer', and 'split'.
    7. Saves the combined DataFrame to a CSV file named 'labels.csv' in the specified output directory.

    Example:
    >>> preprocess_daquar_dataset(output_dir="data/daquar_dataset/")
    Preprocessed data saved to data/daquar_dataset/labels.csv
    """

    # Load train and test data
    train_file_path = os.path.join(output_dir, "train.txt")
    test_file_path = os.path.join(output_dir, "test.txt")

    def read_data(file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()
            questions = [line.strip() for line in lines[::2]]
            answers = [line.strip() for line in lines[1::2]]
            return pd.DataFrame({'question': questions, 'answer': answers})

    train_df = read_data(train_file_path)
    test_df = read_data(test_file_path)

    # Extract image IDs from questions
    train_df['image_id'] = train_df['question'].str.extract(r'image(\d+)')
    train_df['image_id'] = 'image' + train_df['image_id']
    
    test_df['image_id'] = test_df['question'].str.extract(r'image(\d+)')
    test_df['image_id'] = 'image' + test_df['image_id']

    # Add a 'split' column
    train_df['split'] = 'train'
    test_df['split'] = 'test'

    # Combine train and test dataframes
    combined_df = pd.concat([train_df, test_df], ignore_index=True)

    # Reorder columns
    combined_df = combined_df[['question', 'image_id', 'answer', 'split']]
    
    combined_df['question'] = combined_df['question'].str.replace(r'\s*in the image\d+\s*$', '')

    # Save the combined dataframe to a CSV file
    output_file_path = os.path.join(output_dir, "labels.csv")
    combined_df.to_csv(output_file_path, index=False)

    print(f"Preprocessed data saved to {output_file_path}")

    
########### COCO- QA ###########
def get_cocoqa_dataset(output_dir="data/"):
    """
    Downloads and stores the COCO-QA dataset along with COCO images.

    Parameters:
    output_dir (str, optional): The directory where the dataset will be downloaded and uncompressed. Defaults to "data/".

    Example:
    get_cocoqa_dataset(output_dir="data/")

    This example would download and store the COCO-QA dataset and COCO images into the "data/" directory.
    """

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Download and uncompress COCO-QA dataset
        url = "http://www.cs.toronto.edu/~mren/imageqa/data/cocoqa/cocoqa-2015-05-17.zip"
        zip_file_path = os.path.join(output_dir, "cocoqa-2015-05-17.zip")

        response = requests.get(url, stream=True)
        response.raise_for_status()

        with open(zip_file_path, "wb") as zip_file:
            for chunk in response.iter_content(chunk_size=8192):
                zip_file.write(chunk)

        with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
            zip_ref.extractall(output_dir)

        print("COCO-QA dataset downloaded and uncompressed successfully.")

        # Create 'images' directory and download COCO images
        images_dir = os.path.join(output_dir, "images")

        os.makedirs(images_dir, exist_ok=True)

        image_urls = [
            "http://images.cocodataset.org/zips/train2017.zip",
            "http://images.cocodataset.org/zips/val2017.zip"
        ]


        for url in image_urls:
            image_zip_file_path = os.path.join(images_dir, os.path.basename(url))
            response = requests.get(url, stream=True)
            response.raise_for_status()

            with open(image_zip_file_path, "wb") as zip_file:
                for chunk in response.iter_content(chunk_size=8192):
                    zip_file.write(chunk)

            with zipfile.ZipFile(image_zip_file_path, "r") as zip_ref:
                zip_ref.extractall(images_dir)

            # Remove the downloaded zip file for images
            os.remove(image_zip_file_path)


        print("COCO images downloaded and uncompressed successfully.")
        
        # Move all the files from images/train2017 and images/val2017 to images/
        for folder in ["train2017", "val2017"]:
            src_folder = os.path.join(images_dir, folder)
            for file_name in os.listdir(src_folder):
                src_path = os.path.join(src_folder, file_name)
                dst_path = os.path.join(images_dir, file_name)
                shutil.move(src_path, dst_path)

            # Remove the 'train2017/' and 'val2017/' directories
            os.rmdir(src_folder)

    except requests.RequestException as e:
        print(f"Error: {e}")
    except zipfile.BadZipFile as e:
        print(f"Error uncompressing the dataset: {e}")
    finally:
        # Remove the downloaded zip file for COCO-QA dataset
        os.remove(zip_file_path)


def process_cocoqa_data(output_dir="data/"):
    """
    Process the COCO-QA dataset files after downloading.

    This function takes the files in the 'train/' and 'test/' directories of the COCO-QA dataset
    ('answers.txt', 'img_ids.txt', 'questions.txt', 'types.txt'), creates dataframes by concatenating them,
    and saves the resulting dataframes as 'labelsTrain.csv' and 'labelsTest.csv' in their respective directories.
    It then combines these dataframes into a single dataframe 'labels.csv' in the parent folder. The function
    uses unique values in the 'img_ids' column to remove images in the 'images/' directory that are not present
    in the 'img_ids'. The numeric value for each image is obtained from its filename.

    Parameters:
    - output_dir (str, optional): The directory where the dataset is stored. Defaults to "data/".

    Example:
    process_cocoqa_data(output_dir="data/")

    This example assumes that 'get_cocoqa_dataset' has been executed to download and extract the COCO-QA dataset
    and images. It processes the dataset files and organizes them into CSV files while removing unnecessary images.

    """
    
    # Define file paths
    train_labels_path = os.path.join(output_dir, "train/labels.csv")
    test_labels_path = os.path.join(output_dir, "test/labels.csv")
    images_dir = os.path.join(output_dir, "images")
    
    # Read data from train/ and test/ directories
    train_files = os.listdir(os.path.join(output_dir, "train"))
    test_files = os.listdir(os.path.join(output_dir, "test"))

    train_dfs = [pd.read_csv(os.path.join(output_dir, "train", file), header=None, names=[file.split(".")[0]], sep='\t') for file in train_files]
    test_dfs = [pd.read_csv(os.path.join(output_dir, "test", file), header=None, names=[file.split(".")[0]], sep='\t') for file in test_files]

    # Concatenate train/ and test/ dataframes
    labels_train = pd.concat(train_dfs, axis=1)
    labels_test = pd.concat(test_dfs, axis=1)

    # Save train and test dataframes to CSV files
    labels_train.to_csv(train_labels_path, index=False)
    labels_test.to_csv(test_labels_path, index=False)

    print("Train and test dataframes saved successfully.")

    # Concatenate train and test dataframes into a single dataframe
    labels_combined = pd.concat([labels_train, labels_test])
    # Add a new column 'split' based on the original dataset
    labels_combined['split'] = ['train'] * len(labels_train) + ['test'] * len(labels_test)
    
    labels_combined.rename(columns={'img_ids': 'image_id'}, inplace=True)

    # Save combined dataframe to CSV file
    combined_labels_path = os.path.join(output_dir, "labels.csv")
    
    labels_combined.to_csv(combined_labels_path, index=False)

    print("Combined dataframe saved successfully.")

    # Get unique img_ids from the combined dataframe
    unique_img_ids = labels_combined['image_id'].unique()

    # Get all image filenames in the images directory
    all_image_filenames = os.listdir(images_dir)

    # Get the numeric values from the image filenames
    image_numeric_values = [int(filename.split(".")[0]) for filename in all_image_filenames]

    # Get the filenames to be removed from the images directory
    filenames_to_remove = [filename for filename, numeric_value in zip(all_image_filenames, image_numeric_values) if numeric_value not in unique_img_ids]

    # Remove images from the images directory
    for filename in filenames_to_remove:
        file_path = os.path.join(images_dir, filename)
        os.remove(file_path)

    print("Images removed successfully.")

    
########### Fakeddit ###########

def download_fakeddit_files(out_dir='dataset/'):
    """
    Downloads Fakeddit dataset files from Google Drive.

    Parameters:
    - out_dir (str, optional): Output directory where the downloaded files will be stored.
                              Default is 'dataset/'.

    Notes:
    - The function downloads and extracts Fakeddit dataset files from the specified Google Drive links.
    - The files are downloaded as zip files and then extracted, and the zip files are removed after extraction.

    Example:
    ```python
    download_fakeddit_files(out_dir='my_dataset/')
    ```

    Google Drive Links:
    - Test set:   https://drive.google.com/uc?id=1p9EewIKVcFbipVRLZNGYc2JbSC7A0SWv
    - Train set:  https://drive.google.com/uc?id=1XsOkD3yhxgWu8URMes0S9LVSwuFJ4pT6
    - Validation set: https://drive.google.com/uc?id=1Z99QrwpthioZQY2U6HElmnx8jazf7-Kv
    """
    # Create the output directory if it doesn't exist
    os.makedirs(out_dir, exist_ok=True)

    # Define the file URLs
    file_urls = {
        'test': 'https://drive.google.com/uc?id=1p9EewIKVcFbipVRLZNGYc2JbSC7A0SWv',
        'train': 'https://drive.google.com/uc?id=1XsOkD3yhxgWu8URMes0S9LVSwuFJ4pT6',
        'val': 'https://drive.google.com/uc?id=1Z99QrwpthioZQY2U6HElmnx8jazf7-Kv'
    }

    for file_name, file_url in file_urls.items():
        # Download the file
        output_file = os.path.join(out_dir, f'{file_name}.tsv')
        gdown.download(file_url, output_file, quiet=False)

        

def download_full_set_images(out_dir='images/'):
    """
    Downloads and extracts images from the specified Google Drive link.

    Parameters:
    - out_dir (str, optional): Output directory where the extracted images will be stored.
                              Default is 'images/'.

    Notes:
    - The function downloads the 'public_images.tar.bz2' file from the specified Google Drive link
      and extracts its contents into the specified output directory.

    Example:
    ```python
    download_and_extract_images(out_dir='my_images/')
    ```

    Google Drive Link:
    - https://drive.google.com/file/d/1cjY6HsHaSZuLVHywIxD5xQqng33J5S2b/view
    """
    # Create the output directory if it doesn't exist
    os.makedirs(out_dir, exist_ok=True)

    # Define the file URL
    file_url = 'https://drive.google.com/uc?id=1cjY6HsHaSZuLVHywIxD5xQqng33J5S2b'
    
    # Download the tar.bz2 file
    tar_file = os.path.join(out_dir, 'Images.tar.bz2')
    gdown.download(file_url, tar_file, quiet=False)

    # Extract the contents of the tar.bz2 file
    with tarfile.open(tar_file, 'r:bz2') as tar:
        tar.extractall(out_dir)
        
    os.rename(os.path.join(out_dir, 'public_image_set'), os.path.join(out_dir, 'images'))

    # Remove the tar.bz2 file after extraction
    os.remove(tar_file)


def create_stratified_subset_fakeddit(root_path, subset_size, verify=False):
    """
    Create a random stratified subset of the Fakeddit dataset and save it as 'labels.csv'.

    Parameters:
    - root_path (str): The root path to the directory containing the Fakeddit dataset files.
    - subset_size (float): The desired size of the subset, expressed as a fraction between 0 and 1.

    Raises:
    - FileNotFoundError: If the specified root path does not exist.

    Returns:
    None

    The function reads the 'train.tsv', 'test.tsv', and 'val.tsv' files from the specified root path.
    It performs stratified random sampling based on the '2_way_label' column for each dataset.
    The resulting subsets are concatenated into a single dataframe, and a 'split' column is added to
    indicate the original file ('train', 'test', or 'val').
    The resulting dataframe is saved as 'labels.csv' in the root path.

    Example usage:
    ```python
    root_path = 'path/to/your/dataset'
    subset_size = 0.8  # 80% subset size
    create_stratified_subset_fakeddit(root_path, subset_size)
    ```

    Note:
    - Ensure that the dataset files ('train.tsv', 'test.tsv', 'val.tsv') are present in the specified root path.
    - Adjust the subset_size parameter based on the desired size of the subset.

    Dependencies:
    - os
    - pandas as pd
    - train_test_split from sklearn.model_selection
    """
    # Check if the root path exists
    if not os.path.exists(root_path):
        raise FileNotFoundError(f"The root path {root_path} does not exist.")

    # Define the file names
    train_file = os.path.join(root_path, 'train.tsv')
    test_file = os.path.join(root_path, 'test.tsv')
    val_file = os.path.join(root_path, 'val.tsv')

    # Read the datasets
    train_df = pd.read_csv(train_file, sep='\t')
    train_df = train_df[train_df['image_url'].notna()]
    test_df = pd.read_csv(test_file, sep='\t')
    test_df = test_df[test_df['image_url'].notna()]
    val_df = pd.read_csv(val_file, sep='\t')
    val_df = val_df[val_df['image_url'].notna()]
    
    if not(subset_size) or (subset_size >= 1) or (subset_size <= 0):
        train_subset = train_df
        test_subset = test_df
        val_subset = val_df
    else:
        # Stratified sampling based on '2_way_label'
        train_subset, _ = train_test_split(train_df, test_size=1 - subset_size, stratify=train_df['2_way_label'])
        test_subset, _ = train_test_split(test_df, test_size=1 - subset_size, stratify=test_df['2_way_label'])
        val_subset, _ = train_test_split(val_df, test_size=1 - subset_size, stratify=val_df['2_way_label'])

    # Add a 'split' column to indicate the original file
    train_subset['split'] = 'train'
    test_subset['split'] = 'test'
    val_subset['split'] = 'val'

    # Concatenate the subsets
    result_df = pd.concat([train_subset, test_subset, val_subset], ignore_index=True)
    
    result_df['id'] = result_df['id'].astype(str) + '.jpg'
    
    if verify:
        # Get the list of image filenames in the 'images' folder
        images_folder = os.path.join(root_path, 'images')
        image_filenames = [filename for filename in os.listdir(images_folder) if filename.endswith('.jpg')]

        # Filter the 'id' column based on the images in the 'images' folder
        result_df = result_df[result_df['id'].isin(image_filenames)]

    # Save the resulting dataframe to a 'labels.csv' file
    result_df.to_csv(os.path.join(root_path, 'labels.csv'), index=False)


def download_images_from_file(root, labels_name='labels.csv', max_retries=3, delay_seconds=2):
    """
    Download images from URLs specified in a dataframe and save them in an 'images' directory.

    Parameters:
    - root (str): The root path to the directory containing the labels file and where the 'images' directory will be created.
    - labels_name (str, optional): The name of the labels file (default is 'labels.csv').
    - max_retries (int, optional): The maximum number of retries for a failed request (default is 3).
    - delay_seconds (float, optional): The delay between retry attempts in seconds (default is 2).

    Raises:
    - FileNotFoundError: If the specified root path does not exist.

    Returns:
    None

    The function reads the labels file (CSV or TSV) from the specified root path, which should contain columns such as 'hasImage', 'image_url', and 'id'.
    It creates an 'images' directory within the root path and downloads images from the specified URLs, saving them with the 'id' as the filename.
    The progress is displayed using a tqdm progress bar.

    Example usage:
    ```python
    root_path = 'path/to/your/dataset'
    labels_file = 'labels.csv'
    download_images_from_file(root_path, labels_name=labels_file)
    ```

    Note:
    - The labels file should contain columns like 'hasImage', 'image_url', and 'id'.
    - The function uses the 'tqdm' library to display a progress bar during image downloads.

    Dependencies:
    - os
    - pandas as pd
    - numpy as np
    - tqdm
    - urllib.request
    - urllib.error.HTTPError
    - time
    """
    # Check if the root path exists
    if not os.path.exists(root):
        raise FileNotFoundError(f"The root path {root} does not exist.")

    labels_path = os.path.join(root, labels_name)
    images_path = os.path.join(root, "images")

    if labels_path.endswith('.csv'):
        df = pd.read_csv(labels_path)
    else:
        df = pd.read_csv(labels_path, sep="\t")

    df = df.replace(np.nan, '', regex=True)
    df.fillna('', inplace=True)

    pbar = tqdm(total=len(df))

    # Create the output directory if it doesn't exist
    os.makedirs(images_path, exist_ok=True)

    for index, row in df.iterrows():
        if row["hasImage"] == True and row["image_url"] != "" and row["image_url"] != "nan":
            image_url = row["image_url"]
            image_name = row["id"] + ".jpg"
            image_path = os.path.join(images_path, image_name)
            
            if os.path.exists(image_path):
                pbar.update(1)
                continue

            retry_count = 0
            while retry_count < max_retries:
                try:
                    urllib.request.urlretrieve(image_url, image_path)
                    #print(f"Image '{image_name}' downloaded successfully.")
                    break
                except HTTPError as e:
                    if e.code == 429:  # Too Many Requests
                        # print(f"Rate limit exceeded. Retrying in {delay_seconds} seconds...")
                        time.sleep(delay_seconds)
                        retry_count += 1
                    else:
                        # print(f"Failed to download image '{image_name}'. HTTP Error: {e.code}")
                        break
            else:
                print(f"Failed to download image '{image_name}' after {max_retries} retries.")

        pbar.update(1)
    print("done")

    
########### Recipes5k ###########

def download_recipes5k_dataset(out_dir='output/'):
    """
    Downloads and unzips a file from the specified Google Drive link.

    Parameters:
    - out_dir (str, optional): Output directory where the unzipped contents will be stored.
                              Default is 'output/'.

    Notes:
    - The function downloads the file from the specified Google Drive link and extracts its contents
      into the specified output directory.

    Example:
    ```python
    download_and_unzip_file(out_dir='my_output/')
    ```

    Google Drive Link:
    - https://drive.google.com/file/d/11ojDNqjowZIf9RzLWc25_xnoHefDvSBS/view
    """
    # Create the output directory if it doesn't exist
    os.makedirs(out_dir, exist_ok=True)

    # Define the file URL
    file_url = 'https://drive.google.com/uc?id=11ojDNqjowZIf9RzLWc25_xnoHefDvSBS'

    # Download the zip file
    zip_file = os.path.join(out_dir, 'downloaded_file.zip')
    gdown.download(file_url, zip_file, quiet=False)

    # Extract the contents of the zip file
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(out_dir)

    # Remove the zip file after extraction
    os.remove(zip_file)

    
def preprocess_recipes5k(root_dir='dataset/'):
    """
    Preprocesses the Recipes5k dataset and generates a labels.csv file.

    Parameters:
    - root_dir (str, optional): Root directory where the dataset is stored. Default is 'dataset/'.

    Notes:
    - The function reads the ingredients_Recipes5k.txt file and processes the labels and images files
      to create a labels.csv file with additional columns for class and split information.

    Example:
    ```python
    preprocess_recipes5k(root_dir='my_dataset/')
    ```

    Dataset Structure:
    - annotations/<set_split>_images.txt
    - annotations/<set_split>_labels.txt
    - annotations/ingredients_Recipes5k.txt
    """
    # Read ingredients_Recipes5k.txt as a DataFrame
    ingredients_df = pd.read_csv(os.path.join(root_dir, 'annotations', 'ingredients_Recipes5k.txt'), header=None, names=['ingredients'], sep='\t')

    # Initialize an empty DataFrame for labels
    labels_df = pd.DataFrame()

    # Process each set_split (train, test, val)
    for set_split in ['train', 'test', 'val']:
        # Read images and labels files
        images_file = os.path.join(root_dir, 'annotations', f'{set_split}_images.txt')
        labels_file = os.path.join(root_dir, 'annotations', f'{set_split}_labels.txt')

        images_df = pd.read_csv(images_file, header=None, names=['image'], sep='\t')
        labels_indices_df = pd.read_csv(labels_file, header=None, names=['index'], sep='\t')

        # Create class and split columns
        images_df['class'] = images_df['image'].apply(lambda x: x.split('/')[0])
        images_df['split'] = set_split

        # Concatenate images and labels indices DataFrames
        set_split_df = pd.concat([images_df, labels_indices_df], axis=1)

        # Concatenate with the main labels DataFrame
        labels_df = pd.concat([labels_df, set_split_df])

    labels_df = labels_df.set_index('index')
    labels_df.index.name = None

    # Merge with ingredients DataFrame using the index
    final_df = pd.merge(labels_df, ingredients_df, left_index=True, right_index=True)
    final_df = final_df.sort_index()

    # Save the resulting DataFrame as labels.csv
    output_csv_path = os.path.join(root_dir, 'labels.csv')
    final_df.to_csv(output_csv_path, index=False)


########### BRSET ###########
def organize_brset_dataset(output_dir):
    """
    Organizes the downloaded dataset by moving and renaming files and directories.

    This function performs the following tasks:
    1. Moves the 'labels.csv' file from its original location to the 'data/' directory.
    2. Renames the 'fundus_photos' directory to 'images' within the 'data/' directory.
    3. Removes the 'physionet.org' directory and its contents, cleaning up the directory structure.

    Parameters:
    output_dir (str): The path to the root directory where the dataset was downloaded.

    Example:
    organize_brset_dataset("data/")

    This example would move 'labels.csv' to 'data/' and rename 'fundus_photos' to 'images' within 'data/'.
    It would also remove the 'physionet.org' directory and its contents.

    Note: Make sure to call this function after downloading the dataset using 'download_dataset'.
    """

    # Move labels.csv to the data directory
    shutil.move(os.path.join(output_dir, "physionet.org/files/brazilian-ophthalmological/1.0.0/labels.csv"), os.path.join(output_dir, "labels.csv"))

    # Rename fundus_photos to images
    shutil.move(os.path.join(output_dir, "physionet.org/files/brazilian-ophthalmological/1.0.0/fundus_photos"), os.path.join(output_dir, "images"))

    # Remove the physionet.org directory and its contents
    shutil.rmtree(os.path.join(output_dir, "physionet.org"))


def download_dataset(output_dir="data/", url="https://physionet.org/files/brazilian-ophthalmological/1.0.0/"):
    """
    Downloads a dataset from a specified URL and organizes it.

    This function performs the following tasks:
    1. Downloads a dataset from the provided URL using the 'wget' command.
    2. Prompts the user for their PhysioNet username and securely enters their password.
    3. Organizes the downloaded dataset by moving and renaming files and directories using the 'organize_dataset' function.

    Parameters:
    output_dir (str, optional): The directory where the dataset will be downloaded and organized. Defaults to "data/".
    url (str, optional): The URL of the dataset to be downloaded. Defaults to the Brazilian Ophthalmological Dataset on PhysioNet.

    Example:
    download_dataset(output_dir="data/", url="https://physionet.org/files/brazilian-ophthalmological/1.0.0/")

    This example would download the Brazilian Ophthalmological Dataset from the provided URL into the "data/" directory.
    It would prompt the user for their PhysioNet username and securely enter their password.
    After downloading, it would organize the dataset using the 'organize_dataset' function.

    Note: You need to have 'wget' and 'shutil' installed to use this function.
    """

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    username = input("Please provide your physionet's username: ")
    # Prompt the user for the password without displaying it
    password = getpass.getpass("Please provide your physionet's password: ")

    # Run the wget command to download the dataset
    # command = f'wget -r -N -c -np --user {username} --password {password} {url} -P {output_dir}'
    command = f'wget -r -c -np --user {username} --password {password} -nc {url} -P {output_dir}'
    try:
        subprocess.run(command, shell=True, check=True)
        print("Dataset downloaded successfully.")
        
        organize_brset_dataset(output_dir)

    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")


def check_columns(row, columns):
    for column in columns:
        if row[column] != 0:
            return 'abnormal'
    return 'normal'


def get_brset(data_dir, download=False, info=False):
    """
    Reads the dataset CSV file and provides information about the DataFrame.

    Parameters:
    data_dir (str): The directory where the dataset is stored.
    download (bool, optional): Whether to download the dataset if it's not already available. Defaults to False.

    Returns:
    pd.DataFrame: The loaded DataFrame containing dataset information.

    Example:
    df = get_brset("data/", download=True)

    This example would download the dataset if not already available, then load the 'labels.csv' file from the specified directory.
    The resulting DataFrame will contain information about the dataset.

    Note: Make sure to have the 'labels.csv' file in the specified directory.
    """

    if download:
        download_dataset(output_dir=data_dir)

    print(f'loading csv file in {data_dir}/labels.csv')
    df_path = os.path.join(data_dir, 'labels.csv')
    df = pd.read_csv(df_path)

    # Provide information about the DataFrame
    if info:
        print(f"Number of Rows: {df.shape[0]}")
        print(f"Number of Columns: {df.shape[1]}")
        print(f"Column Names: {', '.join(df.columns)}")
        print("\nInfo:")
        print(df.info())

        #print("\nDescription:")
        #print(df.describe())

    columns = ['diabetic_retinopathy', 'macular_edema', 'scar', 'nevus',
               'amd', 'vascular_occlusion', 'hypertensive_retinopathy',
               'drusens', 'hemorrhage', 'retinal_detachment',
               'myopic_fundus', 'increased_cup_disc', 'other'
              ]

    df['normality'] = df.apply(check_columns, args=(columns,),  axis=1)

    return df

def brset_preprocessing(dataset_path, filename='labels.csv', output_filename='labels.csv'):
    # Load the dataset
    df = pd.read_csv(f'{dataset_path}/{filename}')

    # Define the conversion functions
    def convert_sex(sex):
        return 'male' if sex == 1 else 'female' if sex == 2 else 'no sex reported'
    
    def convert_eye(eye):
        return 'right' if eye == 1 else 'left' if eye == 2 else 'no eye reported'
    
    def convert_presence(presence):
        return 'present' if presence == 1 else 'absent'
    
    # Create the 'text' column with conditions
    df['text'] = df.apply(lambda row: (
        f"An image from the {convert_eye(row['exam_eye'])} eye of a {convert_sex(row['patient_sex'])} patient, "
        f"aged {'no age reported' if pd.isnull(row['patient_age']) else str(float(str(row['patient_age']).replace('O', '0').replace(',', '.')))} years, "
        f"{'with no comorbidities reported' if pd.isnull(row['comorbidities']) else 'with comorbidities: ' + row['comorbidities']}, "
        f"{'with no diabetes duration reported' if pd.isnull(row['diabetes_time_y']) or row['diabetes_time_y'] == 'Não' else 'diabetes diagnosed for ' + str(float(str(row['diabetes_time_y']).replace('O', '0').replace(',', '.'))) + ' years'}, "
        f"{'not using insulin' if row['insuline'] == 'no' else 'using insulin'}. "
        f"The optic disc is {convert_presence(row['optic_disc'])}, vessels are {convert_presence(row['vessels'])}, "
        f"and the macula is {convert_presence(row['macula'])}. "
        f"Conditions include macular edema: {convert_presence(row['macular_edema'])}, scar: {convert_presence(row['scar'])}, "
        f"nevus: {convert_presence(row['nevus'])}, amd: {convert_presence(row['amd'])}, vascular occlusion: {convert_presence(row['vascular_occlusion'])}, "
        f"drusens: {convert_presence(row['drusens'])}, hemorrhage: {convert_presence(row['hemorrhage'])}, "
        f"retinal detachment: {convert_presence(row['retinal_detachment'])}, myopic fundus: {convert_presence(row['myopic_fundus'])}, "
        f"increased cup disc ratio: {convert_presence(row['increased_cup_disc'])}, and other conditions: {convert_presence(row['other'])}."
    ), axis=1)

    # Drop all columns except for 'image_id', 'DR_ICDR', and 'text'
    df = df[['image_id', 'DR_ICDR', 'text']]

    # Create DR_2 and DR_3 columns from DR_ICDR
    df['DR_2'] = df['DR_ICDR'].apply(lambda x: 1 if x > 0 else 0)
    df['DR_3'] = df['DR_ICDR'].apply(lambda x: 2 if x == 4 else (1 if x in [1, 2, 3] else 0))

    # Create a 'split' column
    df['split'] = 'train'  # Initialize all as 'train'
    # Stratify split by 'DR_ICDR'
    train_idx, test_idx = train_test_split(df.index, test_size=0.2, stratify=df['DR_ICDR'], random_state=42)
    df.loc[test_idx, 'split'] = 'test'  # Update 'split' for test set

    # Save the processed dataframe to a new CSV file
    df.to_csv(f'{dataset_path}/{output_filename}', index=False)

    print(f"Processed dataset saved as {output_filename} in {dataset_path}")



########### ham10000 ###########
def preprocess_ham10000(path, dir1='HAM10000_images_part_1', dir2='HAM10000_images_part_2', labels='HAM10000_metadata.csv'):
    # Create images directory if not exists
    images_dir = os.path.join(path, 'images')
    os.makedirs(images_dir, exist_ok=True)
    
    # Move images from dir1 and dir2 to images directory
    for directory in [dir1, dir2]:
        full_dir = os.path.join(path, directory)
        for filename in os.listdir(full_dir):
            shutil.move(os.path.join(full_dir, filename), images_dir)
    
    # Load CSV file
    labels_path = os.path.join(path, labels)
    df = pd.read_csv(labels_path)
    
    # Create 'text' column with prompt template
    df['text'] = df.apply(lambda row: f"Patient diagnosed via {row['dx_type']}. Age: {'No data reported' if pd.isnull(row['age']) else int(row['age'])} years. Sex: {row['sex'] if pd.notnull(row['sex']) else 'No data reported'}. Localization: {row['localization'] if pd.notnull(row['localization']) else 'No data reported'}.", axis=1)
    
    # Split data into train and test sets, stratified by 'dx'
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['dx'], random_state=42)
    train_df['split'] = 'train'
    test_df['split'] = 'test'
    
    # Concatenate train and test dataframes
    result_df = pd.concat([train_df, test_df])
    
    # Select relevant columns
    result_df = result_df[['image_id', 'dx', 'text', 'split']]
    
    # Save the resulting CSV
    result_df.to_csv(os.path.join(path, 'labels.csv'), index=False)
    
    return result_df


########### Colombian multi-modal data ###########
def get_satellitedata(data_dir, url= "https://physionet.org/files/multimodal-satellite-data/1.0.0/", download=False, info=False):
    """
    Reads the dataset CSV file and provides information about the DataFrame.

    Parameters:
    data_dir (str): The directory where the dataset is stored.
    download (bool, optional): Whether to download the dataset if it's not already available. Defaults to False.

    Returns:
    pd.DataFrame: The loaded DataFrame containing dataset information.

    Example:
    df = get_satellitedata("data/", download=True)

    This example would download the dataset if not already available, then load the 'labels.csv' file from the specified directory.
    The resulting DataFrame will contain information about the dataset.

    Note: Make sure to have the 'labels.csv' file in the specified directory.
    """

    if download:
        download_dataset(output_dir=data_dir, url=url)

    data_dir = os.path.join(data_dir, "physionet.org/files/multimodal-satellite-data/1.0.0")
    print(data_dir)
    print(f'loading csv file in {data_dir}/metadata.csv')
    df_path = os.path.join(data_dir, 'metadata.csv')
    df = pd.read_csv(df_path)

""" Get epiweek as column name """
def get_epiweek(name):

    # Get week
    week = name.split('/')[1]
    week = week.replace('w','')
    week = int(week)

    # Year
    year = name.split('/')[0]
    year = int(year)

    epiweek = Week(year, week)

    epiweek = str(epiweek)
    epiweek = int(epiweek)

    return epiweek



""" Get labels"""
def read_labels(path, Municipality = None):
    df = pd.read_csv(path)
    if df.shape[1] > 678:
        df = pd.concat([df[['Municipality code', 'Municipality']], df.iloc[:,-676:]], axis=1)
        cols = df.iloc[:, 2:].columns
        new_cols = df.iloc[:, 2:].columns.to_series().apply(get_epiweek)
        df = df.rename(columns=dict(zip(cols, new_cols)))

    if 'Label_CSV_All_Municipality' in path:
        # Get Columns
        df = df[['epiweek', 'Municipality code', 'Municipality', 'final_cases_label']]

        # change epiweek format
        df.epiweek = df.epiweek.apply(get_epiweek)

        # Remove duplicates
        df = df[df.duplicated(['epiweek','Municipality code','Municipality']) == False]

        # Replace Increase, decrease, stable to numerical:
        """
        - Decreased = 0
        - Stable = 1
        - Increased = 2
        """
        df.final_cases_label = df.final_cases_label.replace({'Decreased': 0, 'Stable': 1, 'Increased': 2})

        # Create table
        df = df.pivot(index=['Municipality code', 'Municipality'], columns='epiweek', values='final_cases_label')

        # Reset Index:
        df = df.reset_index()

    if Municipality:

        if type(Municipality) == str:
            if Municipality.isdigit():
                Municipality = int(Municipality)
            else:
                Municipality = int(codes[Municipality])

        df = df[df['Municipality code'] == Municipality]
        df.drop(columns=['Municipality'], inplace=True)
        #df.rename(columns={'Municipality': 'Municipality Code'}, inplace=True)

        df = df.set_index('Municipality code')
        df = df.T

        df.columns.name = None
        df.index.name = None

        df.columns = ['Labels']

        df.index = pd.to_numeric(df.index)

    return df

""" Get Epiweek as int """
def epiweek_from_date(image_date):
    image_date = image_date.replace('/', '-')
    date = image_date.split('-')
    print(date)
    year = ''.join(filter(str.isdigit, date[0]))
    year = int(year)

    # Get month as int

    month = ''.join(filter(str.isdigit, date[1]))
    month = int(month)

    # Get day as int
    day = ''.join(filter(str.isdigit, date[2]))
    day = int(day)

    # Get epiweek:
    if year<1000:
      month_t = month
      year_t = year
      day_t = day
      month = year_t
      day = month_t
      year = day_t
    date = convert_to_date(year, month, day)
    epiweek = str(Week.fromdate(date))
    epiweek = int(epiweek)

    return epiweek


def read_static(path, Municipality = None):
    df = pd.read_csv(path)

    df = df.iloc[:,np.r_[:2, 10:14, 28:53]]

    pop_cols = ['Population2015', 'Population2016', 'Population2017', 'Population2018']
    df['population'] = df[pop_cols].mean(axis=1)
    df.drop(columns=pop_cols, inplace=True)

    df['Date'] = '2016-02-02'

    if 'Municipality code' in df.columns:
        df.rename(columns={'Municipality code':'Municipality Code'}, inplace=True)

    if df['Municipality Code'].dtype == 'int64':
        if type(Municipality) == str:
            if Municipality.isdigit():
                Municipality = int(Municipality)
            else:
                Municipality = int(codes[Municipality])

    if df['Municipality Code'].dtype == 'object':
        if type(Municipality) == int or (type(Municipality) == np.int64):
            Municipality = cities[Municipality]
        elif (type(Municipality) == str) and (Municipality.isdigit()):
            
            Municipality = cities[int(Municipality)]

    if Municipality:
        df = df[df['Municipality Code'] == Municipality]

    df.Date = df.Date.apply(epiweek_from_date)

    df = df.sort_values(by=['Date'])

    df = df.set_index('Date')

    if Municipality:
        df.drop(columns=['Municipality Code','Municipality'], inplace=True)

    df.index.name = None

    return df

def get_temperature_and_precipitation(city, features):

    if type(city) == int or (type(city) == np.int64):
        city = cities[city]
    elif type(city) == str and city.isdigit():
        city = cities[int(city)]

    code = get_code(city)

    # Precipitation
    for col in pd.read_csv(features[0]).columns:
        if code in col:
            column = col
            continue
    precipitation_df = pd.read_csv(features[0])[['LastDayWeek', column]]

    # Temperature
    for col in pd.read_csv(features[1]).columns:
        if code in col:
            column = col
            continue
    temperature_df = pd.read_csv(features[1])[['LastDayWeek', column]]

    # Merge:
    features_df = temperature_df.merge(precipitation_df, how='inner', on='LastDayWeek')

    features_df['LastDayWeek'] = features_df['LastDayWeek'].apply(epiweek_from_date)

    dictionary = {}

    for cols in features_df.columns:
        if 'temperature' in cols:
            dictionary[cols] = 'temperature'
        if 'precipitation' in cols:
            dictionary[cols] = 'precipitation'
    features_df.rename(columns=dictionary, inplace=True)

    features_df = features_df.set_index('LastDayWeek')
    features_df.index.name = None

    return features_df

""" Generate a CSV with the images: """
def read_image(path, crop=True, target_size=(224,224,3), BANDS='RGB', BAND=0):
    if os.path.isdir(path):
        return np.nan
    image_test = io.imread(path)

    if crop:
        x1 = target_size[0] // 2
        x2 = target_size[0] - x1
        y1 = target_size[1] // 2
        y2 = target_size[1] - y1
        x_mid = image_test.shape[0] // 2
        y_mid = image_test.shape[0] // 2

        # selecting part of the image only
        image_arr = image_test[x_mid - x1:x_mid + x2,y_mid - y1:y_mid + y2, :]
        image_arr = image_arr / 255.
    else:
        # Resize the image and normalize values
        image_arr = resize(image_test, (target_size[0], target_size[1]),
                           anti_aliasing=True)

    # If just 3 bands get RGB
    if target_size[2] == 3:
        # RGB - 2, 3, 4
        # CI - 3, 4, 8
        # SWIR- 4, 8, 12
        if BANDS == 'RGB':
            image_arr = image_arr[:, :, [1,2,3]]
        elif BANDS == 'SWIR':
            image_arr = image_arr[:, :, [3,7,11]]
        elif BANDS == 'CI':
            image_arr = image_arr[:, :, [3,4,7]]
    # One band:
    elif target_size[2] == 1:
        image_arr = image_arr[:, :, BAND]
        image_arr = np.expand_dims(image_arr, axis=2)
        image_arr = np.concatenate((image_arr, image_arr, image_arr), axis=2)
    else:
        image_arr = image_arr[:, :, :target_size[2]]

    image_test = np.expand_dims(image_arr, axis=0)

    return image_test

def create_df(images_dir, MUNICIPALITY, target_size=(224, 224, 3), return_paths=None, dataset_path=None):
    print(MUNICIPALITY)
    cities =  {
        76001: "Cali",
        5001: "Medellín",
        50001: "Villavicencio",
        54001: "Cúcuta",
        73001: "Ibagué",
        68001: "Bucaramanga",
        5360: "Itagüí",
        8001: "Barranquilla",
        41001: "Neiva",
        23001: "Montería"
        }
    
    sub_dirs = os.listdir(images_dir)
    sub_dirs = list(map(convert_code, sub_dirs))

    if MUNICIPALITY in sub_dirs:
        MUNICIPALITY = MUNICIPALITY
    else:
        if type(MUNICIPALITY) == int or (type(MUNICIPALITY) == np.int64):
            MUNICIPALITY = cities[MUNICIPALITY]
        elif type(MUNICIPALITY) == str and MUNICIPALITY.isdigit():
            MUNICIPALITY = cities[int(MUNICIPALITY)]
        else:
            MUNICIPALITY = int(codes[MUNICIPALITY])

    images_dir = os.path.join(images_dir, str(MUNICIPALITY))

    out_df = {
        'epiweek':[],
        'image_id':[]
    }

    for image_path in os.listdir(images_dir):
        if image_path.endswith('.tiff'):
            epiweek = epiweek_from_date(image_path)
            full_path = os.path.join(images_dir, image_path)
            index_datasets = full_path.find("datasets")
            desired_path = full_path[index_datasets:]
            out_df['epiweek'].append(epiweek)
            out_df['image_id'].append(desired_path)

    df = pd.DataFrame(out_df)

    df = df.set_index('epiweek')
    df.index.name = None

    if return_paths:
        return df

    # df.image = df.image.apply(read_image, target_size=target_size)
    print(dataset_path)
    dataset_path = os.path.join(dataset_path, 'satellitedata_rgb')
    
    os.makedirs(dataset_path, exist_ok=True)
    
    # Apply the save_image_with_name function to each image path and image name
    for index, row in df.iterrows():
        
        image_id = row['image_id']
        image = read_image(image_id).squeeze(0)
        name_parts = image_id.split('.')
        name_parts[-1] = 'png'
        image_id = '.'.join(name_parts)
        
        path_parts = row['image_id'].split('/')#get image name
        desired_element = path_parts[-2]#get code
        new_name = desired_element+"_"+os.path.basename(row['image_id']).split("_")[1]
        
        imsave(os.path.join(dataset_path, new_name), image)
        
        print("image saved in: ",os.path.join(dataset_path, new_name))

    df = df.dropna()
    return df



def get_dengue_dataset(image_path, labels_path, municipality, temp_prec=False, cases=None, limit=True, static=None, dataset_path=None):

    labels_df = read_labels(path=labels_path, Municipality=municipality)

    if limit:
        labels_df = labels_df[(labels_df.index > 201545) & (labels_df.index < 201901)]

    if  not cases and static and not temp_prec:
        # Only static is True
        static = read_static(path=static, Municipality=municipality)
        labels_df = static.merge(labels_df, how='right', left_index=True, right_index=True)
        labels_df = labels_df.fillna(labels_df.mode().iloc[0])
        features_df = create_df(images_dir= image_path, MUNICIPALITY=municipality, target_size=(224, 224, 3), dataset_path=dataset_path)
        dengue_df = features_df.merge(labels_df, how='inner', left_index=True, right_index=True)
        
        return dengue_df
    else: 
        # Only temp_prec is True
        features_df = get_temperature_and_precipitation(int(municipality), temp_prec)
        dengue_df = features_df.merge(labels_df, how='inner', left_index=True, right_index=True)
        return dengue_df
    
def convert_code(code):
    if code.isdigit():
        return int(code)
    else:
        return code   

def balance_classes(concatenated_df, num_classes):
    concatenated_df['Labels'] = pd.to_numeric(concatenated_df['Labels'], errors='coerce')
    
    # Check if num_classes is 2 or more
    if num_classes == 2:
        median_threshold = concatenated_df['Labels'].median()
        class_labels = (concatenated_df['Labels'] > median_threshold).astype(int)
        
    else:
        # Calculate quantiles based on the number of classes
        quantiles = np.linspace(0, 1, num_classes + 1)[1:-1]  # Exclude 0 and 1 to avoid extremes
        thresholds = concatenated_df['Labels'].quantile(quantiles)
        class_labels = np.digitize(concatenated_df['Labels'], thresholds)
        class_counts = [np.sum(class_labels == i) for i in range(1, num_classes + 1)]
        mean_count = np.mean(class_counts)
        thresholds = [None] * num_classes  # Initialize thresholds with None

        for i in range(num_classes):
            if class_counts[i] > mean_count:
                thresholds[i] = concatenated_df['Labels'][class_labels == i+1].quantile(0.5)
            elif class_counts[i] < mean_count:
                thresholds[i] = concatenated_df['Labels'][class_labels == i+1].quantile(0.5)

        class_labels = np.digitize(concatenated_df['Labels'], thresholds)


    # Update the 'Labels' column with the new class labels
    concatenated_df['Labels'] = class_labels
    return concatenated_df

def satellitedata_preprocessing(output_path='datasets/satellitedata', num_classes = 2, dataset_version="10_municipalities", filename='metadata.csv', output_filename='labels.csv'):     
        
        
        os.makedirs(output_path, exist_ok=True)
        
        dataset_path = os.path.join(os.getcwd(),output_path, "physionet.org/files/multimodal-satellite-data/1.0.0")
        
        labels_path = f'{dataset_path}/{filename}'
        
        temp_prec = ['./precipitation_all.csv', './temperature_all 2.csv']
        
        static = labels_path
        
        cities =  {
        "76001": "Cali",
        "5001": "Medellín",
        "50001": "Villavicencio",
        "54001": "Cúcuta",
        "73001": "Ibagué",
        "68001": "Bucaramanga",
        "5360": "Itagüí",
        "8001": "Barranquilla",
        "41001": "Neiva",
        "23001": "Montería"
        }
        codes = {int(k):v for k,v in cities.items()}
        Municipalities = list(codes.keys())
        image_path = f'{dataset_path}/{dataset_version}/images'
        
        df = [get_dengue_dataset(image_path = image_path, labels_path=labels_path, municipality=Municipality, static=static, dataset_path=output_path) for Municipality in Municipalities]
        new_df = []
        for idx in range(len(df)):
                city = df[idx]
                
                num_classes = num_classes

                city = balance_classes(city, num_classes)

                city['text'] = city.apply(lambda row: (
                f"In a city called {row['Municipality']} with {row['Labels']} Dengue classification, {row['Age0-4(%)']}% of the population is aged 0-4, "
                f"{row['Age5-14(%)']}% aged 5-14, {row['AfrocolombianPopulation(%)']}% are Afro-Colombian, "
                f"and {row['IndianPopulation(%)']}% are of Indian descent."
                ), axis=1)
                columns = ['image_id','text', 'Labels']
                new_df.append(city[columns])
        
        # Unite all dataframes for all cities
        concatenated_df = pd.concat(new_df, ignore_index=True)
        
        print(f"Using {num_classes} classes, data adapted with the following distribution: {concatenated_df.Labels.value_counts()}")
        
        train_val_data, test_data = train_test_split(concatenated_df, test_size=0.2, stratify=concatenated_df['Labels'], random_state=42)
        train_data, val_data = train_test_split(train_val_data, test_size=0.25, stratify=train_val_data['Labels'], random_state=42)  # 0.25 x 0.8 = 0.2

        train_data['split'] = 'train'
        val_data['split'] = 'val'
        test_data['split'] = 'test'

        df_final = pd.concat([train_data, val_data, test_data], ignore_index=True)

        df_final.to_csv(f'{output_path}/{output_filename}', index=False)
        
        print(f"Processed dataset saved as {output_filename} in {dataset_path}")

        return df_final

#####Joslin Diabetes Center Data#####
def joslin_preprocessing(dataset_path, filename='labels.csv', output_filename='labels.csv'):
    # Load the dataset
    df = pd.read_csv(f'{dataset_path}/{filename}')

    # Define the conversion functions
    def convert_sex(sex):
        return 'male' if sex == 'M' else 'female' if sex == 'F' else 'no sex reported'
    
    def convert_eye(eye):
        return 'right' if eye == 'R' else 'left' if eye == 'L' else 'no eye reported'
    
    def convert_presence(presence):
        return 'present' if presence == 1 else 'absent'
    
    # Create the 'text' column with conditions
    # df['text'] = df.apply(lambda row: (
    #     f"An image from the {convert_eye(row['exam_eye'])} eye of a {convert_sex(row['patient_sex'])} patient, "
    #     f"aged {'no age reported' if pd.isnull(row['patient_age']) else str(float(str(row['patient_age']).replace('O', '0').replace(',', '.')))} years, "
    #     f"{'with no comorbidities reported' if pd.isnull(row['comorbidities']) else 'with comorbidities: ' + row['comorbidities']}, "
    #     f"{'with no diabetes duration reported' if pd.isnull(row['diabetes_time_y']) or row['diabetes_time_y'] == 'Não' else 'diabetes diagnosed for ' + str(float(str(row['diabetes_time_y']).replace('O', '0').replace(',', '.'))) + ' years'}, "
    #     f"{'not using insulin' if row['insuline'] == 'no' else 'using insulin'}. "
    #     f"The optic disc is {convert_presence(row['optic_disc'])}, vessels are {convert_presence(row['vessels'])}, "
    #     f"and the macula is {convert_presence(row['macula'])}. "
    #     f"Conditions include macular edema: {convert_presence(row['macular_edema'])}, scar: {convert_presence(row['scar'])}, "
    #     f"nevus: {convert_presence(row['nevus'])}, amd: {convert_presence(row['amd'])}, vascular occlusion: {convert_presence(row['vascular_occlusion'])}, "
    #     f"drusens: {convert_presence(row['drusens'])}, hemorrhage: {convert_presence(row['hemorrhage'])}, "
    #     f"retinal detachment: {convert_presence(row['retinal_detachment'])}, myopic fundus: {convert_presence(row['myopic_fundus'])}, "
    #     f"increased cup disc ratio: {convert_presence(row['increased_cup_disc'])}, and other conditions: {convert_presence(row['other'])}."
    # ), axis=1)

    df['text'] = df.apply(lambda row: (
        f"An image from the {convert_eye(row['LATERALITY'])} eye of a {convert_sex(row['SEX'])} patient, "
        f"aged {'no age reported' if pd.isnull(row['PT_AGE']) else str(float(str(row['PT_AGE']).replace('O', '0').replace(',', '.')))} years, "  
    ), axis=1)

    # Drop all columns except for 'image_id', 'DR_ICDR', and 'text'
    df = df[['ID', 'EYE_DR', 'text']]

    # Create DR_2 and DR_3 columns from DR_ICDR
    df['DR_2'] = df['EYE_DR'].apply(lambda x: 1 if x > 0 else 0)
    df['DR_3'] = df['EYE_DR'].apply(lambda x: 2 if x == 4 else (1 if x in [1, 2, 3] else 0))

    # Create a 'split' column
    df['split'] = 'train'  # Initialize all as 'train'
    # Stratify split by 'EYE_DR'
    train_idx, test_idx = train_test_split(df.index, test_size=0.2, stratify=df['EYE_DR'], random_state=42)
    df.loc[test_idx, 'split'] = 'test'  # Update 'split' for test set

    # Save the processed dataframe to a new CSV file
    df.to_csv(f'{dataset_path}/{output_filename}', index=False)

    print(f"Processed dataset saved as {output_filename} in {dataset_path}")
