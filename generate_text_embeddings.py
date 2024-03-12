
from src.nlp_models import LLAMA
#from src.nlp_models import GPT
import pandas as pd
model = LLAMA(embeddings=True, n_gpu_layers=-1)

model.path = 'datasets/joslin/labels.csv'
column = 'text'
directory = 'Embeddings/joslin'
file = 'text_embeddings.csv'

model.get_embedding_df(column, directory, file)