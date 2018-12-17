"""This file visualises a certain set of pictures by aranging them using PCA
and T-SNE"""

from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from PIL import Image
from sklearn.manifold import TSNE

def get_selection(df, size = 1000, column = 'random', criterium = 'random'):
    """ Returns a sample of data rows from a dataframe according to a chosen
        selection

        Arguments:
            df: dataframe to choose the selection from
            size: size of the returned sample
            column: the name of a column
            criterium: either simple string or list of strings.
                In case of a simple string, all rows where the string is found
                in the specified column are returned.
                In case of a list, all rows where at least one of the strings is
                found will be returned.
                If criterium is set to random an entry of the column is choosen
                by chance and all rows with this entry will be returned.
        Return:
            A dataframe consisting of the rows that meet the given criterium.
        """
    # created dataframe matching selected criteria
    if column == 'random':
        df_sel = df
        if criterium != 'random':
            print("If column = 'random' is chosen the criterium is ignored")
    else:
        if isinstance(criterium, list):
            frames = []
            for searchstr in criterium:
                df_cur = df.loc[df[column].str.find(searchstr) != -1]
                frames.append(df_cur)
            df_sel = pd.concat(frames)
        else:
            if criterium == 'random':
                classes = list(set(df[column]))
                class_sel = classes[np.random.choice(len(classes))]
            else:
                class_sel = criterium
            df_sel = df.loc[df[column].str.find(class_sel) != -1]

    # create random sample of  requested size
    if len(df_sel)< size:
        print('The requested size was longer than the number of matches in the data frame. '
              +'Output size is {0} instead of {1}.'.format(len(df_sel), size))
        size = len(df_sel)
    idx = np.arange(len(df_sel))
    rand = np.random.choice(idx, size=size, replace=False)
    return df_sel.iloc[rand]

def create_visualisation(X, image_path, df_column, save_filepath,
            width = 3000, height = 2200, maxdim = 150):
    """Plots the first two dimensions of a projection to multiple dimensions of
    images by aranging tiles depicting the pictures according to their
    projection.

        Arguments:
            X: The projection, a (n x dim) array where n is the number of
                datapoints and dim the dimension of the projection.
            image_path: Relative path where the images.
            df_column: The column of a pandas dataframe in which the filenames
                of the pictures (Without extension!!) are saved.
            width: integer: width in pixels of the output picture
            height: integer: height in pixels of the output picture.
            maxdim: integer: maximum dimension of a tile in pixels.
            filepath: filepath where the output file shall be saved.

        Return: None. The picture is saved and displayed. """
    #create scatter plot for overview
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.scatter(X[:, 0], -X[:, 1])

    #normalizing to interval between 0 and 1
    X[:,0] = (X[:, 0] - np.min(X[:, 0]))/(np.max(X[:, 0]) - np.min(X[:,0]))
    X[:,1] = (X[:,1] - np.min(X[:, 1]))/(np.max(X[:, 1]) - np.min(X[:,1]))
    full_image = Image.new('RGBA', (width, height))
    print(len(df_sel))
    for i in range(len(df_sel)):
        tile = Image.open(image_path + df_column.iloc[i] + '.jpg')
        xt = X[i,0]
        yt = X[i,1]
        rs = max(1, tile.width/max_dim, tile.height/max_dim)
        tile = tile.resize((int(tile.width/rs), int(tile.height/rs)))#, Image.ANTIALIAS)
        full_image.paste(tile, (int((width-max_dim)*xt), int((height-max_dim)*yt)), mask=tile.convert('RGBA'))
    fig2 = plt.figure(figsize = (10,10))
    ax2 = fig2.add_subplot(111)
    ax2.imshow(full_image)
    full_image.save(save_filepath)
    plt.show()

def load_image(path, target_size = None):
    img = image.load_img(path, target_size = target_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

if __name__ == '__main__':

    # arguments for data selection
    size = 300                              # number of visiualised images
    column = 'artistname'                   # which column to search for criterium
    criterium = ['michelangelo', 'lyonel']  # search criterium (str or list of strs)

    # plot and save settings
    plot_PCA = False
    plot_tSNE = True
    save_filepath = 'michelangelo_feininger_test'

    #  arguments and hyper parameters for PCA and t-SNE
    PCA_components = 50        #t-SNE will be calculated using this projection
    tSNE_components = 2
    early_exaggeration = 30
    perplexity = 10
    learning_rate = 100

    # settings of ouput picture:
    width = 3000
    height = 2200
    max_dim = 150                           # maximum size of displayed tiles

    # arguments
    image_path = "images/"
    df_column_label = '_id'


    # load metadata
    df = pd.read_csv("../data/db.csv",nrows=None,na_values="?")

    # create selection dataframe
    df_sel = get_selection(df, size = size, column = column, criterium = criterium)

    # create feature extractor
    model = VGG16(weights='imagenet', include_top=True)
    feat_extractor = Model(inputs=model.input, outputs=model.get_layer("fc2").output)

    # create feature vectors
    fvs = []
    for i in range(len(df_sel)):        # is slow and could be parallelized
        if i %10 == 0:
            print('extracting feature vector of image {0} of {1}'.format(i+1, len(df_sel)))
        x = load_image(image_path + df_sel['_id'].iloc[i] + '.jpg',
            target_size = feat_extractor.input_shape[1:3])
        fv = feat_extractor.predict(x)[0]
        fvs.append(fv)
    print('done')
    fvs = np.array(fvs)

    # perform PCA to get dimensions down for t-SNE
    modelPCA = PCA(n_components=PCA_components)
    XPCA = modelPCA.fit_transform(fvs)
    # plot PCA visualisation
    if plot_PCA == True:
        save_filepath_PCA = save_filepath + '_PCA.png'
        create_visualisation(XPCA, image_path, def_sel[df_column_label], save_filepath_PCA,
                    width = 3000, height = 2200, maxdim = 150)
    # perform t-SNE
    myTSNE = TSNE(n_components = tSNE_components,
        early_exaggeration = early_exaggeration, perplexity = perplexity,
        learning_rate = learning_rate)
    XTSNE = myTSNE.fit_transform(XPCA)
    # plot t-SNE visualisation
    if plot_tSNE == True:
        save_filepath_tSNE = save_filepath + '_tSNE.png'
        create_visualisation(XPCA, image_path, df_sel[df_column_label], save_filepath_tSNE,
                    width = 3000, height = 2200, maxdim = 150)
