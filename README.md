# Bias in Large Language Models (LLMs) Across Languages

This repository contains code and resources for investigating bias in Large Language Models (LLMs) across multiple languages. The project aims to analyze and mitigate biases present in LLMs in medical text classification across multiple languages.

## Table of Contents

- [Introduction](#introduction)
- [Setup](#setup)
- [Usage](#usage)
- [Data](#data)
- [Analysis](#analysis)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Bias in Large Language Models (LLMs) Across Languages is a research project dedicated to studying and addressing biases that arise in text generation by LLMs when dealing with different languages. This research explores how LLMs may produce biased or stereotypical content in multiple languages and seeks to develop methods to reduce such biases.

## Setup

### Prerequisites

Before running the code, ensure you have the following installed:

- Python 3.9.18
- Required Python packages (specified in `requirements.txt`)

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/dsrestrepo/MIT_LLMs_Language_bias.git
   cd MIT_LLMs_Language_bias
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
This project uses a dataset with medical tests in different languages. Place the required dataset in the `data/` directory.

## Usage

Run main.py from the command line with the desired options. Here's an example command:

```bash
python main.py --csv_file data/YourMedicalTestQuestions.csv --model gpt-3.5-turbo --temperature 0.5 --n_repetitions 3 --reasoning --languages english portuguese french
```

The script accepts the following arguments:

- --csv_file: Specify the path to the CSV file containing your medical test questions.
- --model: Choose the GPT model to use (e.g., gpt-3.5-turbo or gpt-4).
- --temperature: Set the temperature parameter for text generation (default is 0.0).
- --n_repetitions: Define the number of times each question will be asked to the model. This is useful to measure model's consistency.
- --reasoning: Enable reasoning mode to include explanations for responses. If this argument is not provided, the script will only generate responses. This argument increases the number of tokens used and may result in higher costs.
- --languages: Provide a list of languages to process the questions (space-separated). **The name of the questions should match with the column names containing the questions in the CSV file**.

The script will process the questions, generate responses, and save the results in a CSV file.

Alternatively, you can run the jupyter notebook `main.ipynb` to run the code.

We also provide a more customizable option using the class GPT. You can import the class and use it to generate responses from the model, change the prompt, and more. See the file `customized_main.ipynb` for an example.


## Analysis
The analysis results, including bias assessment and mitigation strategies, will be documented in the results/ directory. This is where you can find the results of the test in the LLM across languages.

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
