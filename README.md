# A Wikiart Classifier

**Authors**: [Jan Albrecht](https://github.com/janpfsr), [Matteo Bellitti](https://github.com/mbellitti/), [Mohit Pandey](https://github.com/mohitpandey92)

![PCA](https://github.com/mbellitti/wikiart-classifier/blob/master/data/visualisation/picasso_example120_PCA.png)
_PCA of images of Picasso_



Wikiart-Classifier is an open-source Python-based image classifier.
The idea is to teach a Convolutional NN to recognize the style, genres, and author of an artwork.

This sounds ambitious, some of those concepts are nebulous anyway, and unsurprisingly we do not get accuracy larger than 50% even on the training set.

Still, we think there are many insights to be gained by applying Machine
Learning techniques to this dataset. The header image is a simple application of
PCA to Picasso's work, and clearly there is some structure: for example,
paintings from the blue period are clustered.

This is encouraging, are the similarities in painting of the same school something that's easy to recognize?

This project was the final for the Fall 2018 class
[Machine Learning for Physicists](https://physics.bu.edu/~pankajm/PY895-ML.html)
held at Boston University.

## Dataset
The WikiArt dataset is composed of 152k images, labeled by artist, genre, style and
date of creation, as presented on [WikiArt](https://www.wikiart.org/).

The WikiArt project does not provide a prepackaged dataset, so we used a mix of BeautifulSoup and the all-powerful `wget` to scrape the images and the metadata.

Since the [terms of use](https://www.wikiart.org/en/terms-of-use) are not clear on the
matter of redistribution, if someone is interested in the dataset they should
contact WikiArt first.

## Task considerations
In this dataset, we could run a supervised learning classification task on
- Date the artwork was realized
- Genre (Portrait, Landscape, Symbolic)
- Style (Cubism, Impressionism, Dadaism)
- Author
but some of these are harder than the others.

We trained most on the "genre" label, for two reasons:
- It's the simplest label, with only 59 classes
- There are many more "portraits" than works by any given artist.

It turns out that the most prolific artist (Van Gogh, with 1927 artworks) has
*way* more artworks than usual: the median is just 24 artworks. This means that
if we try to recognize the author of a painting, a few labels will be extremely
easy (the usual suspects: Van Gogh, Renoir, Roerich) but
more than half will have so few datapoints that it's hopeless to learn anything about them.

On the other hand there are 15 *thousand* impressionist paintings, and if
"impressionism" is a concept even remotely well defined, we should be able to
recognize them correctly. The label distribution is better here, with a median
of 176 artworks per style.

# Libraries
We stand on the shoulder of giants: we used the latest version of
Keras, TensorFlow and Python 3.6.2.

The metadata database is implemented as a pandas DataFrame, and processed using the `flow_from_dataframe` function. This is not in the official Keras release yet, a good tutorial can be found [here](https://medium.com/@vijayabhaskar96/tutorial-on-keras-imagedatagenerator-with-flow-from-dataframe-8bd5776e45c1).

# Network Architecture
We were on a tight time schedule, so we used transfer learning to save
computational resources. The basis is VGG16 trained on
[ImageNet](http://www.image-net.org/), followed by a few fully-connected layers and a soft-max classifier.

# Training
We trained the model on the BU [Shared Computing
Cluster](https://www.bu.edu/tech/support/research/computing-resources/scc/),
which has a few nodes equipped with GPUs.

The training test contained 95.5k images, the valdiation set 23k.

We used early stopping to prevent overfitting: after 16 epochs the validation
error started growing again so we interrupted the (Adam) minimization.

![Training](https://github.com/mbellitti/wikiart-classifier/blob/master/src/training.png)

# Testing
The remaining 30k images were used as test set, and we obtained a 49% accuracy on the "style" classification task.

# Misclassification
Looking at a few misclassfied images, it's clear that the main problem is
ambiguity: self-portraits and portraits are commonly mistaken for each other,
but the NN recognizes it's confused by assigning roughly equal weights to the
two possibilities.

![A self-portrait misclassified as portrait.](https://github.com/mbellitti/wikiart-classifier/blob/master/data/portrait.png)

For this particular example Prob(self-por.) = 0.67 and Prob(por.) = 0.72

Some other artworks are even more ambiguous:

![A still life](https://github.com/mbellitti/wikiart-classifier/blob/master/data/still.png)

this is officially a "still life", but our model classifies it with high confidence as "abstract". Can we really blame it?

Overall, we think the model is performing well, and if we want to improve the misclassification errors we need to think mode deeply about the topic.

# Clustering
This dataset lends itself to unsupervised learning tasks, too: the header image
was one example, and we played with t-SNE and a few other artists to see what
features are captured by clustering.
We used the pre-prediction layer of VGG16 for feature extraction and applied PCA and t-SNE.
Interestingly, using our trained network as a feature extractor did not only not improve the result but made them significantly worse compared to using the dense layers of VGG16. The following to pictures show PCA applied to pictures py picasso using VGG and our trained network:
![PCA with VGG16.](https://github.com/mbellitti/wikiart-classifier/blob/master/data/visualisation/vgg_picasso_300_PCA.png)
_PCA of images of Picasso using full VGG16_
![PCA with VGG16.](https://github.com/mbellitti/wikiart-classifier/blob/master/data/visualisation/om_picasso_300_PCA.png)
_PCA of images of Picasso using our trained network_

Although, we couldn't get the visualisation to clearly separate different authors, there is still a visible divide between the works of picasso and rubens in this case:
![tSNE on VGG16](https://github.com/mbellitti/wikiart-classifier/blob/master/data/visualisation/vgg_picassorubens_300_ea30_tSNE.png)
_tSNE of images of Picasso and PEter Paul Rubens_
