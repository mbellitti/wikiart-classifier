# A Wikiart Classifier

**Authors**: Jan Albrecht, Matteo Bellitti, [Mohit Pandey](https://github.com/mohitpandey92)

Wikiart-Classifier is an open-source Python-based image classifier. It classifies artwork into different genres (?) using Deep Neural Networks. 

We have achieved 45 % accuracy on test set with a total number of 150,000 images and XXX number of labels on them. If we had randomly guessed the label of an image, then we get an accuracy of 1 %. (I have assumed number of labels is 100).

Images to be added:
- Plot of accuracy as a function of epochs.


![PCA](https://github.com/mbellitti/wikiart-classifier/blob/visualisation/src/picasso_example500_PCA.png?raw=true "Title")
_PCA of images of Picasso_


![tSNE](https://github.com/mbellitti/wikiart-classifier/blob/visualisation/src/michelangelo_feininger_test_tSNE.png?raw=true "Title")
_tSNE of images of Michelangelo_

Footnote:
In principle, we have data to run classification task on
- Century the artwork was realized
- Field (Painting, Sculpture, Photography)
- Style (Cubism, Impressionism, Dadaism)
- Author



This was done as part of final project for the Fall 2018 class
[Machine Learning for Physicists](https://physics.bu.edu/~pankajm/PY895-ML.html) held at Boston University.


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


# **Task consideration**

In the end we trained most on the "style" of a painting than on any other label, simply because there are many "mythological paintings" and many "landscapes" but very few paintings by any given artist.

It turns out that the most prolific artist (Van Gogh, with 1927 artworks) has way more artworks than usual: the median is just 24 artworks. This means that if we try to recognize the author of a painting, a few labels will be extremely easy (the usual suspects: Van Gogh, Renoir, Roerich... You know him, right?) and more than half will have so few datapoints that it's unlikely we will ever figure them out.

On the other hand there are 15 thousand impressionist paintings, and if "impressionism" is a concept even remotely well defined, we should be able to guess them correctly. The distribution is better here, with a median of 176 artworks per style.


# **Network Architecture**

We use Deep Neural Network to classify images. Specifically, our network is Convolutional Neural Network (CNN), whose first layer is a pre-trained layer (VGG16) followed by dense sets and at the end a softmax classifier.



# **What we did**
Since the WikiArt project does not provide a precompiled dataset, we built our own database by scraping from [WikiArt](https://www.wikiart.org/)  the images and the corresponding metadata (labels) we are interested in. Our strategy was based on that fact that WikiArt has a listing of all authors in alphabetical order. (Should we talk more about specific difficulties we faced while scraping metadata? )


Once we had all the images and the corresponding metadata, we built a Pandas dataframe with columns containing filenames and corresponding metadata like genre and style.

Since we didn't have great amount of data and computational powers compared to companies like Google, we built a CNN using the concept of transfer learning. We use a pre-trained neural network VGG16 as our first layer. The weights of VGG16 was trained on ImageNet, which is a collection of about one million images. These are very deep models, which generalise well to other datasets. That's why we used ImageNet as our initial layers, which helped our CNN in learning general features. The output of these layers was fed to dense layers, which helped our CNN in learning features specific to our imageset. We trained our CNN on GPUs and find that with as less as 5 epochs, our CNNs achieve the optimal accuracy.

We used VGG16 to extract features and then used Principal Component Analysis and tSNE to find the clustering in images of a specific artist. (@Jan, do you want to add more here?)



# **What Python packages we used**
- [Pandas](https://pandas.pydata.org/) to create a dataframe with image file names and corresponding metadata
- [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/) to scrape images
- [Keras](https://keras.io/) for image manipulation, loading pre-trained VGG16 and training CNNs

(should we mention tensorflow?)


# **What we are offering**

- Image-scraper that works for WikiArt but can be extended for other websites too
- A GPU trained CNN on wikiart (?)
- Data scraped from WikiArt (?)--Probably we might get in trouble if we put Wikiart data on our github since Wikiart actually "owns" the data.


