# A Wikiart Classifier

**Authors**: Jan Albrecht, Matteo Bellitti, [Mohit Pandey](https://github.com/mohitpandey92)

Wikiart-Classifier is an open-source Python-based image classifier. It classifies artwork into different genre (?) using Deep Neural Networks (for details about architecture, checkout this).  

We have achieved 45 % accuracy on test set with XXX number of images and XXX number of labels on them.

*Images*
-Plot of accuracy as a function of epochs. 
-Jan's PCA image

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
* [Network Architecture](#Network-Architecture)
* [What we did](#What-we-did)
* [What we are offering](#What-we-are-offering)


# **Data** 

We have XXX number of images which have so many labels.  The source of the images and metadata is
[WikiArt](https://www.wikiart.org/). 

 

# **Network Architecture**

We use Deep Neural Network to classify images. Specifically, our network is Convolutional Neural Network (CNN), whose first layer is a pre-trained layer (VGG16) followed by dense sets and at the end a softmax classifier.



# **What we did**

We scraped images from [WikiArt](https://www.wikiart.org/) using [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/). Once we had all the images, we built a CNN using tranfer learning. We use a pre-trained neural network as our first layer.


# **What packages we used**
We used python 3.6
- [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/) to scrape images
- keras for image manipulation, loading pre-trained VGG16 and training CNNs 
 
tenserflow (should we mention tenserflow?)


# **What we are offering**

- Image-scraper that works for WikiArt but can be extended for other websites too
- A GPU trained CNN on wiki-art (?)
- Data scraped from WikiArt (?)
