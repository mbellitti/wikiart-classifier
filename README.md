# A Wikiart Classifier

**Authors**: Jan Albrecht, Matteo Bellitti, [Mohit Pandey](https://github.com/mohitpandey92)

![PCA](https://github.com/mbellitti/wikiart-classifier/blob/visualisation/src/picasso_example500_PCA.png?raw=true "Title")
_PCA of images of Picasso_

![tSNE](https://github.com/mbellitti/wikiart-classifier/blob/visualisation/src/michelangelo_feininger_test_tSNE.png?raw=true "Title")
_tSNE of images of Michelangelo_

Wikiart-Classifier is an open-source Python-based image classifier.
The idea is to teach a Convolutional NN to recognize the style, genres, and author of an artwork.

This sounds ambitious, since those concepts are nebulous anyway, and unsurprisingly we do not get accuracy larger than 50% even on the training set.

Still, we think there are many insights to be gained by applying Machine Learning techniques to this dataset.

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

We trained most on the "style" label, for two reasons:
- We think it's more interesting than the genre
- There are many "mythological paintings" and than works by any given artist.

It turns out that the most prolific artist (Van Gogh, with 1927 artworks) has
*way* more artworks than usual: the median is just 24 artworks. This means that
if we try to recognize the author of a painting, a few labels will be extremely
easy (the usual suspects: Van Gogh, Renoir, Roerich) but
more than half will have so few datapoints that it's hopeless to learn anything about them.

On the other hand there are 15 *thousand* impressionist paintings, and if
"impressionism" is a concept even remotely well defined, we should be able to
recognize them correctly. The label distribution is better here, with a median
of 176 artworks per style.


# **Contents**
--------
* [Installation](#Installation)
* [Data](#Data)
* [Network Architecture](#Network-Architecture)
* [What we did](#What-we-did)
* [What Python packages we used](What-Python-packages-we-used)
* [What we are offering](#What-we-are-offering)


# **Installation**
You should install latest version of Keras, Tensorflow and Python 3. After the installation is finished, you can download all the files from Github repository to your local directory.

# **Data**

We have XXX number of images which have XXX labels.  The source of the images and metadata is
[WikiArt](https://www.wikiart.org/).


# **Network Architecture**
We were on a tight time schedule, so we used transfer learning to save
computational resources. The basis is VGG16 trained on
[ImageNet](http://www.image-net.org/), followed by a few fully-connected layers and a soft-max classifier.

Once we had all the images and the corresponding metadata, we built a Pandas dataframe with columns containing filenames and corresponding metadata like genre and style.

# Training
We trained the model on the BU [Shared Computing Cluster](https://www.bu.edu/tech/support/research/computing-resources/scc/), which has a few nodes equipped with GPUs.

We used early stopping to prevent overfitting: after 16 epochs the validation error started growing again so we interrupted the (Adam) minimization.

![Training](https://github.com/mbellitti/wikiart-classifier/blob/master/src/training.png)

# Clustering
We used VGG16 to extract features and then used Principal Component Analysis and tSNE to find the clustering in images of a specific artist. (@Jan, do you want to add more here?)

# **What Python packages we used**
- [Pandas](https://pandas.pydata.org/) to create a dataframe with image file names and corresponding metadata
- [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/) to scrape images
- [Keras](https://keras.io/) for image manipulation, loading pre-trained VGG16 and training CNNs
