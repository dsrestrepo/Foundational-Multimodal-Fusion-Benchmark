# Multimodal Data Fusion Using Foundational Models.

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

This repository contains a framework for Multimodal Data Fusion using Foundational Models. The framework allows you to extract, preprocess and combine data from different sources and modalities using the most powerful vision and language models

## Table of Contents

- [Introduction](#introduction)
- [Setup](#setup)
- [Usage](#usage)
- [Data](#data)
- [Analysis](#analysis)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Domestic violence prediction is a critical task that can benefit from multimodal data fusion. This framework leverages state-of-the-art foundational models to combine satellite imagery and social media data for enhanced prediction accuracy. The framework is flexible and can be adapted to other multimodal data fusion tasks.

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

This project uses 5 datasets. You'll find instructions and code about to extract each dataset in `get_datasets.ipynb`:

1. Gender Violence Dataset: A dataset of internet data such as social media or google searches, and satellite images to predict gender violence. The codes can be used to extract a dataset for other tasks. The codes to extrac the dataset are avaibale in: `datasets/violence_prediction`.

* Satellite: To download the satellite images go to `datasets/violence_prediction/Satellite`. There you'll find the satellite extractor, this code uses the [Sentinel Hub API](https://www.sentinel-hub.com/develop/api/). Take into account that the satellite extractor requires the coordinates of the Region of Interes (ROI). You can use the file `Coordinates/get_coordinates.ipynb` to generate the ROI of your specific location. There is also a `DataAnalysis.ipynb` to assess the quality of the images.
* Metadata: The labels are located in the directory `datasets/violence_prediction/Metadata`. The labels were downloaded from open public data sources through the number of police reports of domestic violence reported in Colombia  from January 1, 2010 to August 28, 2023. You can find information about the data sources in the `data_sources.txt`. Use the `get_dataset.ipynb` to preprocess and merge the data sources, and the `Data_Analysis.ipynb` to run a data analysis.

2. [DAQUAR Dataset](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/vision-and-language/visual-turing-challenge#c7057): DAQUAR (Dataset for Question Answering on Real-world images) dataset was created for the purpose of advancing research in visual question answering (VQA). It consists of indoor scene images, each accompanied by sets of questions related to the scene's content. The dataset serves as a benchmark for training and evaluating models in understanding images and answering questions about them.

3. [COCO-QA Dataset](https://www.cs.toronto.edu/~mren/research/imageqa/data/cocoqa/): The COCO-QA (COCO Question-Answering) dataset is designed for the task of visual question-answering. It is a subset of the COCO (Common Objects in Context) dataset, which is a large-scale dataset containing images with object annotations. The COCO-QA dataset extends the COCO dataset by including questions and answers associated with the images. Each image in the COCO-QA dataset is accompanied by a set of questions and corresponding answers.

4. [Fakeddit Dataset](https://fakeddit.netlify.app/): Fakeddit is a large-scale multimodal dataset for fine-grained fake news detection. It consists of over 1 million samples from multiple categories of fake news, including satire, misinformation, and fabricated news. The dataset includes text, images, metadata, and comment data, making it a rich resource for developing and evaluating fake news detection models.

5. [Recipes5k Dataset](http://www.ub.edu/cvub/recipes5k/): The Recipes5k dataset comprises 4,826 recipes featuring images and corresponding ingredient lists, with 3,213 unique ingredients simplified from 1,014 by removing overly-descriptive particles, offering a diverse collection of alternative preparations for each of the 101 food types from Food101, meticulously balanced across training, validation, and test splits. The dataset addresses intra- and inter-class variability, extracted from Yummly with 50 recipes per food type.

## Usage

1. Get the dataset: Use the notebook `get_datasets.ipynb`. Functions and code to extract and preprocess each dataset were created.



## Analysis


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
