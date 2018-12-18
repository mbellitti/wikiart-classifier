# A Wikiart Classifier

**Authors**: Jan Albrecht, Matteo Bellitti, [Mohit Pandey](https://github.com/mohitpandey92)


# Overview

Wikiart-Classifier is an open-source Python-based image classifier. It classifies artwork into different genre using Deep Neural Networks (for details about architecture, checkout this).  

We have achieved 45 % accuracy on test set with XXX number of images and XXX number of labels on them.


Footnote:
In principle, we have data to run classificiation task on
- Century the artwork was realized
- Field (Painting, Sculpture, Photography)
- Style (Cubism, Impressionism, Dadaism)
- Author



This was done as part of final project for the Fall 2018 class
[Machine Learning for Physicists](https://physics.bu.edu/~pankajm/PY895-ML.html) held at BU.


# **Contents**
--------
* [Data](#Data)
* [Network Architecture](#Network Architecture)
* [What we did](#What we did)
* [What we are offering](#What we are offering)


# Data 

We have XXX number of images which have so many labels.  The source of the images and metadata is
[WikiArt](https://www.wikiart.org/). 

 

# Network Architecture

We use Deep Neural Network to classify images. Specifically, our network is Convolutional Neural Network, whose first layer is a pre-trained layer (VGG16) followed by dense sets and at the end a softmax classifier.



#What we did





# What we are offering

- Image-scraper that works for WikiArt but can be extended for other websites too
- A GPU trained CNN on wiki-art
- Data scraped from WikiArt (?)
