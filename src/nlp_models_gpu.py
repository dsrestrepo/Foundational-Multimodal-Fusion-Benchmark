""" Evaluate Medical Tests Classification in LLMS """
## Setup
#### Load the API key and libaries.
import os
import re

import json
import pandas as pd
import argparse
import subprocess
import numpy as np

# Create a class to handle the GPT API
class GPT:
    # build the constructor
    def __init__(self, model='gpt-3.5-turbo', temperature=0.0, n_repetitions=1, reasoning=False, languages=['english', 'portuguese'], path='data/Portuguese.csv', max_tokens=500):
        
        import openai
        from dotenv import load_dotenv, find_dotenv
        _ = load_dotenv(find_dotenv()) # read local .env file
        openai.api_key  = os.environ['OPENAI_API_KEY']

        self.path = path
        self.model = model
        self.temperature = temperature
        self.n_repetitions = n_repetitions if n_repetitions > 0 else 1
        self.reasoning = reasoning
        self.languages = languages
        self.max_tokens = max_tokens
        
        self.delimiter = "####"
        self.responses = ['A', 'B', 'C', 'D']
        self.extra_message = ""

        if self.reasoning:
            self.output_keys = ['response', 'reasoning']
        else:
            self.output_keys = ['response']

        self.update_system_message()


    def update_system_message(self):
        """
        Update the system message based on the current configuration.
        """

        if self.reasoning:
            self.system_message = f"""
            You will be provided with medical queries in this languages: {", ".join(self.languages)}. \
            The medical query will be delimited with \
            {self.delimiter} characters.
            Each question will have {len(self.responses)} possible answer options.\
            provide the letter with the answer and a short sentence answering why the answer was selected. \
            {self.extra_message}

            Provide your output in json format with the \
            keys: {", ".join(self.output_keys)}.

            Responses: {", ".join(self.responses)}.

            """
        else:
            self.system_message = f"""
            You will be provided with medical queries in this languages: {", ".join(self.languages)}. \
            The medical query will be delimited with \
            {self.delimiter} characters.
            Each question will have {len(self.responses)} possible answer options.\
            provide the letter with the answer.
            {self.extra_message}

            Provide your output in json format with the \
            keys: {", ".join(self.output_keys)}.

            Responses: {", ".join(self.responses)}.
            
            """

    # function to change the delimiter
    def change_delimiter(self, delimiter):
        """ Change the delimiter """
        self.delimiter = delimiter        
        self.update_system_message()

    # function to change the responses
    def change_responses(self, responses):
        self.responses = responses
        self.update_system_message()
    
    def change_output_keys(self, output_keys):
        self.output_keys = output_keys
        self.update_system_message()
    
    def add_output_key(self, output_key):
        self.output_keys.append(output_key)
        self.update_system_message()

    def change_languages(self, languages):
        self.languages = languages
        self.update_system_message()
    
    def add_extra_message(self, extra_message):
        self.extra_message = extra_message
        self.update_system_message()
    
    def change_system_message(self, system_message):
        self.system_message = system_message

    def change_reasoning(self, reasoning=None):
        if type(reasoning) == bool:
            self.reasoning = reasoning
        else:
            if reasoning:
                print(f'Reasoning should be boolean. Changing reasoning from {self.reasoning} to {not(self.reasoning)}.')        
            self.reasoning = False if self.reasoning else True
        
        if self.reasoning:
            self.output_keys.append('reasoning')
            # remove duplicates
            self.output_keys = list(set(self.output_keys))
        else:
            try:
                self.output_keys.remove('reasoning')
            except:
                pass
        self.update_system_message()

    #### Template for the Questions
    def generate_question(self, question):

        user_message = f"""/
        {question}"""
        
        messages =  [  
        {'role':'system', 
        'content': self.system_message}, 
        {'role':'user', 
        'content': f"{self.delimiter}{user_message}{self.delimiter}"},  
        ] 
        
        return messages
    

    def get_embedding(self, text):
        from openai import OpenAI
        client = OpenAI()

        text = text.replace("\n", " ")

        return client.embeddings.create(input = [text], model=self.model)['data'][0]['embedding']

    def get_embedding_df(self, column, directory, file):
        df = pd.read_csv(self.path)
        df["embeddings"] = df[column].apply(lambda x: self.get_embedding(x))

        os.makedirs(directory, exist_ok=True) 
        df.to_csv(f"{directory}/{file}", index=False)

        
    #### Get the completion from the messages
    def get_completion_from_messages(self, prompt):
        
        messages = self.generate_question(prompt)

        try:        
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature, 
                max_tokens=self.max_tokens,
                request_timeout=10
            )
        except:
            response = self.get_completion_from_messages(prompt)
            return response

        response = response.choices[0].message["content"]

        # Convert the string into a JSON object
        response = json.loads(response)
    
        return response

    def llm_language_evaluation(self, save=True):
        ### Questions from a csv file:
        df = pd.read_csv(self.path)

        ### Evaluate the model in question answering per language:
        responses = {}
        for key in self.output_keys:
            responses[key] = {}
            for language in self.languages:
                responses[key][language] = [[] for n in range(self.n_repetitions)]

        for row in range(df.shape[0]):
            print('*'*50)
            print(f'Question {row+1}: ')
            for language in self.languages:
                print(f'Language: {language}')                   
                question = df[language][row]                    
                print('Question: ')
                print(question)                        
                for n in range(self.n_repetitions): 
                    print(f'Test #{n}: ')
                    response = self.get_completion_from_messages(question)
                    print(response)
                    for key in self.output_keys:
                        # Append to the list:
                        responses[key][language][n].append(response[key])
            print('*'*50)

        ### Save the results in a csv file:
        for language in self.languages:
            if self.n_repetitions == 1:
                for key in self.output_keys:
                    df[f'{key}_{language}'] = responses[key][language][0]
            else:
                for n in range(self.n_repetitions):
                    for key in self.output_keys:
                        df[f'{key}_{language}_{n}'] = responses[key][language][n]
        if save:
            if not os.path.exists('responses'):
                os.makedirs('responses')
            if self.n_repetitions == 1:
                df.to_csv(f"responses/{self.model}_Temperature{str(self.temperature).replace('.', '_')}.csv", index=False)
            else:
                df.to_csv(f"responses/{self.model}_Temperature{str(self.temperature).replace('.', '_')}_{self.n_repetitions}Repetitions.csv", index=False)

        return df

    
    
# Create a class to handle the LLAMA 2
class LLAMA:
    # build the constructor
    def __init__(self, model='Llama-2-7b', embeddings=False, temperature=0.0, n_repetitions=1, reasoning=False, languages=['english', 'portuguese'], path='data/Portuguese.csv', max_tokens=2048, n_ctx=2048, verbose=False, n_gpu_layers=None, echo=True):
        
        self.embeddings = embeddings
        self.n_gpu_layers = n_gpu_layers
        
        self.model = model
        model_path = self.download_hugging_face_model(model)
        
        from llama_cpp import Llama
        if self.n_gpu_layers:
            self.llm = Llama(model_path=model_path, embedding=self.embeddings, verbose=verbose, n_gpu_layers=self.n_gpu_layers, echo = True)
            print(f"Testing: self.n_gpu_layers: {self.n_gpu_layers}")
        
        else:
            self.llm = Llama(model_path=model_path, embedding=self.embeddings, verbose=verbose)
                
        self.path = path
        
        self.temperature = temperature
        self.n_repetitions = n_repetitions if n_repetitions > 0 else 1
        self.reasoning = reasoning
        self.languages = languages
        self.max_tokens = max_tokens
        
        self.delimiter = "####"
        self.responses = ['A', 'B', 'C', 'D']
        self.extra_message = ""

        if self.reasoning:
            self.output_keys = ['response', 'reasoning']
        else:
            self.output_keys = ['response']

        self.update_system_message()


    def update_system_message(self):
        """
        Update the system message based on the current configuration.
        """

        if self.reasoning:
            self.system_message = f"""
            You will be provided with medical queries in this languages: {", ".join(self.languages)}. \
            The medical query will be delimited with \
            {self.delimiter} characters.
            Each question will have {len(self.responses)} possible answer options.\
            provide the letter with the answer and a short sentence answering why the answer was selected. \
            {self.extra_message}

            Provide your output in json format with the \
            keys: {", ".join(self.output_keys)}. Make sure to always use the those keys, do not modify the keys.
            Be very careful with the resulting JSON file, make sure to add curly braces, quotes to define the strings, and commas to separate the items within the JSON.

            Responses: {", ".join(self.responses)}.
            """
        else:
            self.system_message = f"""
            You will be provided with medical queries in this languages: {", ".join(self.languages)}. \
            The medical query will be delimited with \
            {self.delimiter} characters.
            Each question will have {len(self.responses)} possible answer options.\
            {self.extra_message}

            Provide your output in json format with the \
            keys: {", ".join(self.output_keys)}. Make sure to always use the those keys, do not modify the keys.
            Be very careful with the resulting JSON file, make sure to add curly braces, quotes to define the strings, and commas to separate the items within the JSON.

            Responses: {", ".join(self.responses)}.
            """
    def download_and_rename(self, url, filename):
        """Downloads a file from the given URL and renames it to the given new file name.

        Args:
            url: The URL of the file to download.
            new_file_name: The new file name for the downloaded file.
        """

        os.makedirs(os.path.dirname(filename), exist_ok=True)

        print(f'Downloading the weights of the model: {url} ...')
        subprocess.run(["wget", "-q", "-O", filename, url])
        print(f'Done!')
        
    def download_hugging_face_model(self, model_version='Llama-2-7b'):
        if model_version not in ['Llama-2-7b', 'Llama-2-13b', 'Llama-2-70b']:
            raise ValueError("Options for Llama model should be 7b, 13b or 70b")

        MODEL_URL = {
            'Llama-2-7b': 'https://huggingface.co/TheBloke/Llama-2-7b-Chat-GGUF/resolve/main/llama-2-7b-chat.Q8_0.gguf', 
            'Llama-2-13b': 'https://huggingface.co/TheBloke/Llama-2-13B-chat-GGUF/resolve/main/llama-2-13b-chat.Q8_0.gguf', 
            'Llama-2-70b': 'https://huggingface.co/TheBloke/Llama-2-70B-chat-GGUF/resolve/main/llama-2-70b-chat.Q5_0.gguf'
        }

        MODEL_URL = MODEL_URL[model_version]

        model_path = f'Models/{model_version}.gguf'

        if os.path.exists(model_path):
            confirmation = input(f"The model file '{model_path}' already exists. Do you want to overwrite it? (yes/no): ").strip().lower()
            if confirmation != 'yes':
                print("Model installation aborted.")
                return model_path

        self.download_and_rename(MODEL_URL, model_path)

        return model_path

    # function to change the delimiter
    def change_delimiter(self, delimiter):
        """ Change the delimiter """
        self.delimiter = delimiter        
        self.update_system_message()

    # function to change the responses
    def change_responses(self, responses):
        self.responses = responses
        self.update_system_message()
    
    def change_output_keys(self, output_keys):
        self.output_keys = output_keys
        self.update_system_message()
    
    def add_output_key(self, output_key):
        self.output_keys.append(output_key)
        self.update_system_message()

    def change_languages(self, languages):
        self.languages = languages
        self.update_system_message()
    
    def add_extra_message(self, extra_message):
        self.extra_message = extra_message
        self.update_system_message()
    
    def change_system_message(self, system_message):
        self.system_message = system_message

    def change_reasoning(self, reasoning=None):
        if type(reasoning) == bool:
            self.reasoning = reasoning
        else:
            if reasoning:
                print(f'Reasoning should be boolean. Changing reasoning from {self.reasoning} to {not(self.reasoning)}.')        
            self.reasoning = False if self.reasoning else True
        
        if self.reasoning:
            self.output_keys.append('reasoning')
            # remove duplicates
            self.output_keys = list(set(self.output_keys))
        else:
            try:
                self.output_keys.remove('reasoning')
            except:
                pass
        self.update_system_message()

    #### Template for the Questions
    def generate_question(self, question):

        user_message = f"""/
        {question}"""
        
        messages =  [  
        {'role':'system', 
        'content': self.system_message}, 
        {'role':'user', 
        'content': f"{self.delimiter}{user_message}{self.delimiter}"},  
        ] 
        
        return messages

    def get_embedding(self, text):
        
        if  self.index % 500 == 0:
            print(f'{self.index} Embeddings generated!')
            
        self.index += 1 

        text = text.replace("\n", " ")

        text = text.replace("\n", " ")
        text = text.replace("                                 ", "")
        text = text.replace("FINAL REPORT", " ")
        try:
            embed = self.llm.create_embedding(input = [text])['data'][0]['embedding']
        except:
            embed = np.nan

        return embed

    def get_embedding_df(self, column, directory, file):
        self.index = 0
        df = pd.read_csv(self.path)
        df["embeddings"] = df[column].apply(lambda x: self.get_embedding(x))

        os.makedirs(directory, exist_ok=True) 
        df.to_csv(f"{directory}/{file}", index=False)


    
    #### Get the completion from the messages
    def get_completion_from_messages(self, prompt):
        
        messages = self.generate_question(prompt)

        response = self.llm.create_chat_completion(
            messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens)
        
        self.llm.set_cache(None)

        response = response['choices'][0]['message']["content"]        

        # Convert the string into a JSON object
        try:
            # Use regular expressions to extract JSON
            json_pattern = r'\{.*\}'  # Match everything between '{' and '}'
            match = re.search(json_pattern, response, re.DOTALL)
            response = match.group()

            # Define a regex pattern to identify unquoted string values
            pattern = r'("[^"]*":\s*)([A-Za-z_][A-Za-z0-9_]*)'
            # Use a lambda function to add quotes to unquoted string values
            response = re.sub(pattern, lambda m: f'{m.group(1)}"{m.group(2)}"', response)
            
            # Convert
            response = json.loads(response)
        except:
            print(f'Error converting respose to json: {response}')
            print('Generating new response...')
            response = self.get_completion_from_messages(prompt)
            return response
        
        if self.reasoning:
            # Iterate through the keys of the dictionary
            for key in list(response.keys()):
                if 'reas' in key.lower():
                    # Update the dictionary with the new key and its corresponding value
                    response['reasoning'] = response.pop(key)
            
        return response


    def llm_language_evaluation(self, save=True):

        ### Questions from a csv file:
        df = pd.read_csv(self.path)

        ### Evaluate the model in question answering per language:
        responses = {}
        for key in self.output_keys:
            responses[key] = {}
            for language in self.languages:
                responses[key][language] = [[] for n in range(self.n_repetitions)]

        for row in range(df.shape[0]):
            print('*'*50)
            print(f'Question {row+1}: ')
            for language in self.languages:
                print(f'Language: {language}')                   
                question = df[language][row]                    
                print('Question: ')
                print(question)                        
                for n in range(self.n_repetitions): 
                    print(f'Test #{n}: ')
                    response = self.get_completion_from_messages(question)
                    print(response)
                    for key in self.output_keys:
                        # Append to the list:
                        responses[key][language][n].append(response[key])
            print('*'*50)
        
        

        ### Save the results in a csv file:
        for language in self.languages:
            if self.n_repetitions == 1:
                for key in self.output_keys:
                    df[f'{key}_{language}'] = responses[key][language][0]
            else:
                for n in range(self.n_repetitions):
                    for key in self.output_keys:
                        df[f'{key}_{language}_{n}'] = responses[key][language][n]
        if save:
            if not os.path.exists('responses'):
                os.makedirs('responses')
            if self.n_repetitions == 1:
                df.to_csv(f"responses/{self.model}_Temperature{str(self.temperature).replace('.', '_')}.csv", index=False)
            else:
                df.to_csv(f"responses/{self.model}_Temperature{str(self.temperature).replace('.', '_')}_{self.n_repetitions}Repetitions.csv", index=False)

        return df





from transformers import AutoTokenizer, AutoModel
import torch


## Hugging face Models
class HuggingFaceEmbeddings:
    """
    A class to handle text embedding generation using a Hugging Face pre-trained transformer model.
    This class loads the model, tokenizes the input text, generates embeddings, and provides an option 
    to save the embeddings to a CSV file.

    Args:
        model_name (str, optional): The name of the Hugging Face pre-trained model to use for generating embeddings. 
                                    Default is 'sentence-transformers/all-MiniLM-L6-v2'.
        path (str, optional): The path to the CSV file containing the text data. Default is 'data/file.csv'.
        save_path (str, optional): The directory path where the embeddings will be saved. Default is 'Models'.
        device (str, optional): The device to run the model on ('cpu' or 'cuda'). If None, it will automatically detect 
                                a GPU if available; otherwise, it defaults to CPU.

    Attributes:
        model_name (str): The name of the Hugging Face model used for embedding generation.
        tokenizer (transformers.AutoTokenizer): The tokenizer corresponding to the chosen model.
        model (transformers.AutoModel): The pre-trained model loaded for embedding generation.
        path (str): Path to the input CSV file.
        save_path (str): Directory where the embeddings CSV will be saved.
        device (torch.device): The device on which the model and data are processed (CPU or GPU).

    Methods:
        get_embedding(text):
            Generates embeddings for a given text input using the pre-trained model.

        get_embedding_df(column, directory, file):
            Reads a CSV file, computes embeddings for a specified text column, and saves the resulting DataFrame 
            with embeddings to a new CSV file in the specified directory.

    Example:
        embedding_instance = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', 
                                                   path='data/products.csv', save_path='output')
        text_embedding = embedding_instance.get_embedding("Sample product description.")
        embedding_instance.get_embedding_df(column='description', directory='output', file='product_embeddings.csv')

    Notes:
        - The Hugging Face model and tokenizer are downloaded from the Hugging Face hub.
        - The function supports large models and can run on either GPU or CPU, depending on device availability.
        - The input text will be truncated and padded to a maximum length of 512 tokens to fit into the model.
    """
    
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2', path='data/file.csv', save_path=None, device=None, quantization=None, access_token=None):
        """
        Initializes the HuggingFaceEmbeddings class with the specified model and paths.

        Args:
            model_name (str, optional): The name of the Hugging Face pre-trained model. Default is 'sentence-transformers/all-MiniLM-L6-v2'.
            path (str, optional): The path to the CSV file containing text data. Default is 'data/file.csv'.
            save_path (str, optional): Directory path where the embeddings will be saved. Default is 'Models'.
            device (str, optional): Device to use for model processing. Defaults to 'cuda' if available, otherwise 'cpu'.
            quantization (str, optional): The type of quantization to use. Can be 'fp16', 'int8', or 'int4'. Defaults to None.
            access_token (str, optional): An optional access token for loading private models from the Hugging Face hub.
        """
        self.model_name = model_name
        self.quantization = quantization
        
        # Load the Hugging Face tokenizer from a pre-trained model
        if access_token:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token)
            # Load model with the desired quantization
            if self.quantization == 'fp16':
                print("Loading model with fp16 quantization...")
                self.model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16, token=access_token)
            elif self.quantization == 'int8':
                print("Loading model with 8-bit quantization...")
                from transformers import BitsAndBytesConfig
                bnb_config = BitsAndBytesConfig(load_in_8bit=True)
                self.model = AutoModel.from_pretrained(model_name, quantization_config=bnb_config, token=access_token)
            elif self.quantization == 'int4':
                print("Loading model with 4-bit quantization...")
                from transformers import BitsAndBytesConfig
                bnb_config = BitsAndBytesConfig(load_in_4bit=True)
                self.model = AutoModel.from_pretrained(model_name, quantization_config=bnb_config, token=access_token)
            else:
                print("Loading model without quantization...")
                self.model = AutoModel.from_pretrained(model_name)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            # Load model with the desired quantization
            if self.quantization == 'fp16':
                print("Loading model with fp16 quantization...")
                self.model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16)
            elif self.quantization == 'int8':
                print("Loading model with 8-bit quantization...")
                from transformers import BitsAndBytesConfig
                bnb_config = BitsAndBytesConfig(load_in_8bit=True)
                self.model = AutoModel.from_pretrained(model_name, quantization_config=bnb_config)
            elif self.quantization == 'int4':
                print("Loading model with 4-bit quantization...")
                from transformers import BitsAndBytesConfig
                bnb_config = BitsAndBytesConfig(load_in_4bit=True)
                self.model = AutoModel.from_pretrained(model_name, quantization_config=bnb_config)
            else:
                print("Loading model without quantization...")
                self.model = AutoModel.from_pretrained(model_name)
        
        if 'Llama-3.1' in model_name:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.path = path
        self.save_path = save_path or 'Models'
        
        # Define device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            

        # Move model to the specified device if not quantized
        if self.quantization in ['int8', 'int4']:
            print(f"Quantized model ({self.quantization}) is already on the correct device.")
            # Check the device of the quantized model
            first_param_device = next(self.model.parameters()).device
            print(f"Quantized model is on device: {first_param_device}")
        else:
            print(f"Using device: {self.device}")
            self.model.to(self.device)
            print(f"Model moved to device: {self.device}")
            
        print(f"Model: {model_name}")
        
        
    def get_embedding(self, text):
        """
        Generates embeddings for a given text using the Hugging Face model.

        Args:
            text (str): The input text for which embeddings will be generated.

        Returns:
            np.ndarray: A numpy array containing the embedding vector for the input text.
        """
        ### Tokenize the input text using the Hugging Face tokenizer
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        
        # Move the inputs to the device
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        
        with torch.no_grad():
            # Generate the embeddings using the Hugging Face model from the tokenized input
            outputs = self.model(**inputs)
        
        # Extract the embeddings from the model output, send to cpu and return the numpy array
        # The resulting tensor should have shape [batch_size, hidden_size]
        embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()[0]
        
        return embeddings

    def get_embedding_df(self, column, directory, file):
        # Load the CSV file
        df = pd.read_csv(self.path)
        # Generate embeddings for the specified column using the `get_embedding` method
        df["embeddings"] = df[column].apply(lambda x: self.get_embedding(x).tolist())
        
        # Save the DataFrame with the embeddings to a new CSV file in the specified directory
        os.makedirs(directory, exist_ok=True)
        df.to_csv(f"{directory}/{file}", index=False)