# Multimodal Data Fusion Using Embeddings.

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

This repository contains a framework for Multimodal Data Fusion using Foundational Models' Embeddings, and and Embedding Alignment method. The framework allows you to extract, pre-process and align embedding data from different sources and modalities using the most powerful vision and language models making a more efficient use of resources and data.

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
git clone https://github.com/dsrestrepo/Foundational-Multimodal-Fusion-Benchmark.git
cd Foundational-Multimodal-Fusion-Benchmark
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

## Data

This project uses 8 datasets. You'll find instructions and code about to extract each dataset in `get_datasets.ipynb`:

### General Datasets:

1. [DAQUAR Dataset](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/vision-and-language/visual-turing-challenge#c7057): DAQUAR (Dataset for Question Answering on Real-world images) dataset was created for the purpose of advancing research in visual question answering (VQA). It consists of indoor scene images, each accompanied by sets of questions related to the scene's content. The dataset serves as a benchmark for training and evaluating models in understanding images and answering questions about them.

2. [COCO-QA Dataset](https://www.cs.toronto.edu/~mren/research/imageqa/data/cocoqa/): The COCO-QA (COCO Question-Answering) dataset is designed for the task of visual question-answering. It is a subset of the COCO (Common Objects in Context) dataset, which is a large-scale dataset containing images with object annotations. The COCO-QA dataset extends the COCO dataset by including questions and answers associated with the images. Each image in the COCO-QA dataset is accompanied by a set of questions and corresponding answers.

3. [Fakeddit Dataset](https://fakeddit.netlify.app/): Fakeddit is a large-scale multimodal dataset for fine-grained fake news detection. It consists of over 1 million samples from multiple categories of fake news, including satire, misinformation, and fabricated news. The dataset includes text, images, metadata, and comment data, making it a rich resource for developing and evaluating fake news detection models.

4. [Recipes5k Dataset](http://www.ub.edu/cvub/recipes5k/): The Recipes5k dataset comprises 4,826 recipes featuring images and corresponding ingredient lists, with 3,213 unique ingredients simplified from 1,014 by removing overly-descriptive particles, offering a diverse collection of alternative preparations for each of the 101 food types from Food101, meticulously balanced across training, validation, and test splits. The dataset addresses intra- and inter-class variability, extracted from Yummly with 50 recipes per food type.

### Medical Datasets:

5. [BRSET Dataset](https://physionet.org/content/brazilian-ophthalmological/1.0.0/): The Brazilian Multilabel Ophthalmological Dataset (BRSET) stands as a pioneering initiative aimed at bridging the gap in ophthalmological datasets, particularly for under-represented populations in low and medium-income countries. This comprehensive dataset encompasses 16,266 images from 8,524 Brazilian patients, incorporating a wide array of data points including demographics, anatomical parameters of the macula, optic disc, and vessels, along with quality control metrics such as focus, illumination, image field, and artifacts.

6. [HAM10000 dataset](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T) : The MNIST: HAM10000 dataset is a large collection of dermatoscopic images from different populations, acquired and stored by the Department of Dermatology at the Medical University of Vienna, Austria. It consists of 10,015 dermatoscopic images which can serve as a training set for academic machine learning purposes in tasks like skin lesion analysis and classification, specifically focusing on the detection of melanoma.

7. [A Multi-Modal Satellite Imagery Dataset for Public Health Analysis in Colombia](https://physionet.org/content/multimodal-satellite-data/1.0.0/) : The Multi-Modal Satellite Imagery Dataset in Colombia integrates economic, demographic, meteorological, and epidemiological data. It comprises 12,636 high-quality satellite images from 81 municipalities between 2016 and 2018, with minimal cloud cover. Its applications include deforestation monitoring, education indices forecasting, water quality assessment, extreme climatic event tracking, epidemic illness addressing, and precision agriculture optimization. We'll use it shortly.

8.[MIMIC CXR](https://physionet.org/content/mimic-cxr/2.0.0/#files-panel) : The MIMIC-CXR (Medical Information Mart for Intensive Care, Chest X-Ray) dataset is a large, publicly available collection of chest radiographs with associated radiology reports. It was developed by the MIT Lab for Computational Physiology and provides an extensive resource for training and evaluating machine learning models in the field of medical imaging, particularly in automated radiograph interpretation and natural language processing for clinical narratives.

## Usage

1. **Get the Dataset:**
    - Utilize the `get_datasets.ipynb` notebook to acquire the dataset. Functions and code for extraction and preprocessing are provided.

2. **Extract the Embeddings:**
    - **Text Embeddings:**
        - For extracting text embeddings, use models supporting OpenAI API like GPT-3.5, GPT-4, or models compatible with the llama cpp package such as LLAMA 2 7B, LLAMA 2 13B, LLAMA 2 70B, or Mistral 7B. Refer to `generate_text_embeddings.ipynb` for the process, in this case we used LLAMA 2 70B.
    - **Image Embeddings:**
        - Choose from 21 available pre-trained computer vision models to extract image embeddings. Check out `generate_image_embeddings.ipynb` for the process, in this example we used Dino V2 Base.

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
                - Metrics: Accuracy, F1-score, training time, and inference time measured for each model.
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

        - **Embedding Alignment:**
            - See the `Alignment/` for examples on how to align the embeddings. The notebooks provides a method to align the embeddings. The method aligns the embeddings in the same space, and reduces the modality gap improving the data fusion without requiring full model training or additional fine-tuning.

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
