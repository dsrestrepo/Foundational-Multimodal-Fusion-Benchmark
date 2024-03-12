
from src.nlp_models_gpu import LLAMA
#from src.nlp_models import GPT
import pandas as pd
model = LLAMA(embeddings=True)

model.path = 'datasets/joslin/labels.csv'
column = 'text'
directory = 'Embeddings/joslin'
file = 'text_embeddings.csv'

model.get_embedding_df(column, directory, file)