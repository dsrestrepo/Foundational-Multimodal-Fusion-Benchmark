# Multimodal Data Fusion Using Foundational Models.

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

This repository contains a framework for Multimodal Data Fusion using Foundational Models, with a specific focus on Domestic Violence Prediction. The framework allows you to combine satellite images and social media data to enhance the accuracy of domestic violence prediction models.

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

- Python 3.9.18
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

4. Set up your OpenAI API key:

Create a `.env` file in the root directory.

Add your OpenAI API key to the `.env` file:

```makefile
OPENAI_API_KEY=your_api_key_here
```

Make sure you have a valid OpenAI API key to access the language model.

## Data
This project uses a dataset of social media data, and satellite images extracted using the codes and data avaiable in `experiments/violence_prediction` in each directory.

## Usage



## Analysis


## Contributing
Contributions to this research project are welcome. To contribute:

1. Fork the repository.
2. Create a new branch for your feature or research.
3. Make your changes.
4. Create tests.
5. Submit a pull request.

We encourage the community to join our efforts to understand and mitigate bias in LLMs across languages.

## License
This project is licensed under the MIT License.


In this updated README, the focus is on bias analysis in LLMs across languages. You can customize it further to include specific details about your data sources, analysis methodologies, and mitigation strategies related to bias in LLMs across different languages.

## Contact

For any inquiries or questions regarding this project, please feel free to reach out:

- **Email:** davidres@mit.edu
