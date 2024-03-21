from src.vlm_models import CLIP, BLIP2, LLAVA
from src.classifiers_base import preprocess_df
import pandas as pd
import os

model_name = 'clip'

if model_name.lower() == 'clip':
    print('Creating Instance of CLIP model')
    model = CLIP()
elif model_name.lower() == 'blip2':
    print('Creating Instance of BLIP 2 model')
    model = BLIP2()
elif model_name.lower() == 'llava':
    print('Creating Instance of LLAVA model')
    model = LLAVA()
else:
    raise NotImplementedError('The model should be clip, blip2 or llava')

import os

current_directory = os.getcwd()
print("Current directory:", current_directory)

batch_size = 24
dataset = 'joslin'
image_col = 'ID'
text_col = 'text'
output_dir = f'Embeddings_vlm/{dataset}'
output_file = 'embeddings_clip.csv'

images_dir = 'images/'
labels = 'labels.csv'

images_path = os.path.join(current_directory, "datasets", dataset, images_dir)
labels_path = os.path.join(current_directory, "datasets", dataset, labels)

df = preprocess_df(df=pd.read_csv(labels_path), image_columns=str(image_col), images_path=images_path)

# import pdb;pdb.set_trace()

model.get_embeddings(dataframe=df, batch_size=batch_size, image_col_name=image_col, text_col_name=text_col, output_dir=output_dir, output_file=output_file)