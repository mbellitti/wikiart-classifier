# A Wikiart Classifier

**Authors**: Jan Albrecht, Matteo Bellitti, Mohit Pandey

This page is dedicate to the final project for the Fall 2018 class
[Machine Learning for Physicists](https://physics.bu.edu/~pankajm/PY895-ML.html) held at BU.

# Overview

We have built a classifier that classifies artwork into different genre (like Abstract, Landscape, Portrait). In principle, we have data to run classificiation task on

- Century the artwork was realized
- Field (Painting, Sculpture, Photography)
- Style (Cubism, Impressionism, Dadaism)
- Author

The idea is to classify artworks into different styles, genres, and author using Deep Neural Networks.

This is probably very ambitious. 

On this kind of data, in order of increasing data availability, we could
try to classify:


# What we are offering

- Image-scraper that works for WikiArt but can be extended for other websites too
- A GPU trained CNN on wiki-art
- Data scraped from WikiArt (?)




# Data 

## Data sources

The source of the images and metadata is
[WikiArt](https://www.wikiart.org/). The WikiArt project does not
provide a precompiled dataset, so we will need to build our own by
scraping the images and the labels we are interested in.

At the very least we will need to collect the following features:

- Image
- Century
- Field

it would also be interesting to deal with

- Style
- Date (difficulty: often approximate and not in standard format)
- Genre (difficulty: paintings in the same genre can be visually
  very different)
- Author 

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

We use Deep Neural Network to classify images. Specifically, our network is Convolutional Neural Network, whose first layer is a pre-trained layer (VGG16) followed by dense sets and at the end a softmax classifier.

