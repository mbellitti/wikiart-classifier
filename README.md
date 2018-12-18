# A Wikiart Classifier

**Authors**: Jan Albrecht, Matteo Bellitti, Mohit Pandey

This page is about the final project for the Fall 2018 class
[Machine Learning for Physicists](https://physics.bu.edu/~pankajm/PY895-ML.html) held at BU.

# Overview

The idea is to teach an NN to recognize the style, genres, and author of an artwork.

This sounds ambitious, since those concepts are nebulous anyway, and unsurprisingly we do not get accuracy larger than 50% even on the training set.

On this kind of data, in order of increasing data availability, we could
try to classify:

- Century the artwork was realized
- Field (Painting, Sculpture, Photography)
- Genre (Abstract, Landscape, Portrait)
- Style (Cubism, Impressionism, Dadaism)
- Author

## Dataset

The dataset was compiled by scraping [WikiArt](https://www.wikiart.org/). The WikiArt project does not
provide a prepackaged dataset, so we used a

## Task considerations
In the end we trained most on the "style" of a painting than on any other label,
simply because there are many "mythological paintings" and many "landscapes" but
very few paintings by any given artist.

It turns out that the most prolific
artist (Van Gogh, with 1927 artworks) has *way* more artworks than usual: the median is just 24 artworks. This means that if we try to recognize the author of a painting, a few labels will be extremely easy (the usual suspects: Van Gogh, Renoir, Roerich... You know him, right?) and more than half will have so few datapoints that it's unlikely we will ever figure them out.

On the other hand there are 15 *thousand* impressionist paintings, and if
"impressionism" is a concept even remotely well defined, we should be able to
guess them correctly. The distribution is better here, with a median of 176
artworks per style.


## Preprocessing

The labels must be properly formatted and deduplicated, but most of
this work has already been done by the people on WikiArt.

The images will need more preprocessing:

- Image resolution should be made homogeneous
- Pixel values need to be normalized

## Data Augmentation

Many artists have less than 20 artworks, so if we want to perform
with acceptable results on the Author classification task we need
to exploit data augmentation as much as possible.

The situation is less dire for the top 10 Genres, but we still risk
overfitting.

# Network Architecture

We are dealing with image data, so we will heavily use
Convolutional Neural Network, but the detailed network architecture
has not been defined yet.

# Timeline

There is not an official deadline yet. The official end of Finals is on 2018-21-12.
