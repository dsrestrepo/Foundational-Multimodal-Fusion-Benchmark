# Multimodal Data Fusion Using Embeddings.

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

This repository contains a framework for Multimodal Data Fusion using Foundational Models' Embeddings. The framework allows you to extract, preprocess and combine data from different sources and modalities using the most powerful vision and language models making a more efficient use of resources and data.

## Table of Contents

- [Introduction](#introduction)
- [Setup](#setup)
- [Usage](#usage)
- [Data](#data)
- [Analysis](#analysis)
- [Contributing](#contributing)
- [License](#license)

## Introduction

This framework leverages state-of-the-art foundational models to combine multimodal data for enhanced predictions. The framework is flexible and can be adapted to other multimodal data fusion tasks.

## Setup

### Prerequisites

Before running the code, ensure you have the following installed:

- Python 3.8.15
- Required Python packages (specified in `requirements.txt`)

### Installation

1. Clone the repository:

```bash
git clone https://github.com/dsrestrepo/Foundational-Multimodal-Fusion-Framework.git
cd Foundational-Multimodal-Fusion-Framework
```

2. Create a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Set up your OpenAI API (Optional) key if you'll use GPT as foundational model:

Create a `.env` file in the root directory.

Add your OpenAI API key to the `.env` file:

```makefile
OPENAI_API_KEY=your_api_key_here
```
Make sure you have a valid OpenAI API key to access the language model.

5. Set up your Sentinel Hub APIs to get the Satellite Images:

You'll get those in your profile in your sentinell hub account.

```makefile
CLIENT_ID = your_client_id
CLIENT_SECRET = your_client_secret
```


## Data

This project uses 4 datasets. You'll find instructions and code about to extract each dataset in `get_datasets.ipynb`:

1. [DAQUAR Dataset](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/vision-and-language/visual-turing-challenge#c7057): DAQUAR (Dataset for Question Answering on Real-world images) dataset was created for the purpose of advancing research in visual question answering (VQA). It consists of indoor scene images, each accompanied by sets of questions related to the scene's content. The dataset serves as a benchmark for training and evaluating models in understanding images and answering questions about them.

2. [COCO-QA Dataset](https://www.cs.toronto.edu/~mren/research/imageqa/data/cocoqa/): The COCO-QA (COCO Question-Answering) dataset is designed for the task of visual question-answering. It is a subset of the COCO (Common Objects in Context) dataset, which is a large-scale dataset containing images with object annotations. The COCO-QA dataset extends the COCO dataset by including questions and answers associated with the images. Each image in the COCO-QA dataset is accompanied by a set of questions and corresponding answers.

3. [Fakeddit Dataset](https://fakeddit.netlify.app/): Fakeddit is a large-scale multimodal dataset for fine-grained fake news detection. It consists of over 1 million samples from multiple categories of fake news, including satire, misinformation, and fabricated news. The dataset includes text, images, metadata, and comment data, making it a rich resource for developing and evaluating fake news detection models.

4. [Recipes5k Dataset](http://www.ub.edu/cvub/recipes5k/): The Recipes5k dataset comprises 4,826 recipes featuring images and corresponding ingredient lists, with 3,213 unique ingredients simplified from 1,014 by removing overly-descriptive particles, offering a diverse collection of alternative preparations for each of the 101 food types from Food101, meticulously balanced across training, validation, and test splits. The dataset addresses intra- and inter-class variability, extracted from Yummly with 50 recipes per food type.

## Usage

1. **Get the Dataset:**
    - Utilize the `get_datasets.ipynb` notebook to acquire the dataset. Functions and code for extraction and preprocessing are provided.

2. **Extract the Embeddings:**
    - **Text Embeddings:**
        - For extracting text embeddings, use models supporting OpenAI API like GPT-3.5, GPT-4, or models compatible with the llama cpp package such as LLAMA 2 7B, LLAMA 2 13B, LLAMA 2 70B, or Mistral 7B. Refer to `generate_text_embeddings.ipynb` for the process, in this case we used LLAMA 2 70B.
    - **Image Embeddings:**
        - Choose from 21 available pretrained computer vision models to extract image embeddings. Check out `generate_image_embeddings.ipynb` for the process, in this example we used Dino V2 Base.

3. **Run the Experiments:**
    - For each dataset, two notebooks are available:
        - **Embeddings Model:**
            - Use `embed_model_dataset_name.ipynb` to test embeddings. The model supports early and late fusion approaches.
                - *Early Fusion Approach:* Concatenate text and image embeddings, pass through a layer with 256 neurons, 0.2 dropout, ReLu Activation, and Batch Normalization. Connect to a classification layer.
                - *Late Fusion Approach:* Process text and image embeddings with two layers each (128 neurons, ReLu activation, 0.2 dropout, and BatchNorm). Concatenate outputs and connect to a classification layer.
            - **Training Details:**
                - Batch size: 64
                - Workers: 2
                - Optimizer: AdamW with a learning rate of 0.001
                - Loss Function: Binary Cross-Entropy for multilabel and binary classification; Cross-Entropy for multiclass classification.
                - Metrics: Accuracy, training time, and inference time measured for each model.
                - Additional option of class weights is available.

        - **Baseline Model:**
            - In `full_model_dataset_name.ipynb`, a baseline model utilizes a BERT transformer for text data and a ViT-based architecture for image data. Information from both backbones is passed to two classification heads with the same configurations as the embedding models.
            - **Training Details:**
                - Batch size: 64
                - Workers: 2
                - Optimizer: AdamW with a learning rate of 0.001
                - Loss Function: Binary Cross-Entropy for multilabel and binary classification; Cross-Entropy for multiclass classification.
                - Metrics: Accuracy, training time, and inference time measured for each model.
                - Additional option of class weights is available.

## Contributing
Contributions to this research project are welcome. To contribute:

1. Fork the repository.
2. Create a new branch for your feature or research.
3. Make your changes.
4. Create tests.
5. Submit a pull request.


## License
This project is licensed under the MIT License.


## Contact

For any inquiries or questions regarding this project, please feel free to reach out:

- **Email:** davidres@mit.edu
