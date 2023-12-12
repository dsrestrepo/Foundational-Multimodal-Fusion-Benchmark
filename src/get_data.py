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


