Recipes5k is a dataset for ingredients recognition with 4,826 unique recipes composed of an image and the corresponding list of ingredients. It contains a total of 3,213 unique ingredients (10 per recipe on average). Each recipe represents an alternative way to prepare one of the 101 food types contained in Food101. Hence, it captures at the same time the intra- and inter-class variability of cooking recipes. The nearly 50 alternative recipes belonging to each of the 101 classes were divided in train, val and test splits in a balanced way.

The main problem found when dealing with the 3,213 raw ingredients was that many of them were sub-classes (e.g. 'sliced tomato' or 'tomato sauce') of more general versions of themselves (e.g. 'tomato'). For this reason, we applied a simple removal of overly-descriptive particles like 'sliced' or 'sauce', resulting in a simplified version of 1,014 ingredients.

In order to download our data, we selected the web platform Yummly (http://www.yummly.com/), we downloaded around 50 recipes per each food type present in Food101. Each recipe contains an image and its associated unique list of ingredients.

## Contents

Following we describe the files available in each of the folders:

* images/<query_class>/<img_name>.jpg - each of the images of the dataset, where <query_class> is the query name used for retrieving the recipes data from Yummly, and <img_name> is a unique name for the given image. Note that NOT ALL images belonging to a certain <query_class> will belong to the corresponding class in Food101.

* annotations/<set_split>_images.txt - list of images belonging to each of the data splits, where <set_split> can be either 'train', 'val' or 'test'.

* annotations/<set_split>_labels.txt - list of indices for each of the images in <set_split>_images.txt. Each index points to the corresponing line in ingredients_Recipes5k.txt

* annotations/ingredients_Recipes5k.txt - comma separated file that contains, in each line, the list of ingredients present in a certain image of the dataset.

* annotations/classes_Recipes5k.txt - list of unique name identifiers for each of the images.

## Citation

If you use this dataset for any purpose, please, do not forget to cite the following paper:

```
Marc Bolaños, Aina Ferrà and Petia Radeva. "Food Ingredients Recognition through Multi-label Learning" In Proceedings of the 3rd International Workshop on Multimedia Assisted Dietary Management (ICIAP Workshops), 2017. Pre-print: https://arxiv.org/abs/1707.08816
```

## Contact

If you have any doubt or proposal, please, do not hesitate to contact the first author.

Marc Bolaños
marc.bolanos@ub.edu
http://www.ub.edu/cvub/marcbolanos/
