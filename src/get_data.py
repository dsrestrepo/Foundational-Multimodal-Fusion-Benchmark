import os
import requests
import tarfile
import zipfile
import shutil
import pandas as pd


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

    # Save the combined dataframe to a CSV file
    output_file_path = os.path.join(output_dir, "labels.csv")
    combined_df.to_csv(output_file_path, index=False)

    print(f"Preprocessed data saved to {output_file_path}")

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
