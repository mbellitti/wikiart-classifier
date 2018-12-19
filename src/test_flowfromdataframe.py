from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers, optimizers
import pandas as pd
import numpy as np


traindf=pd.read_csv('../data/db.csv', nrows = 5)#, na_values = '?')
datagen=ImageDataGenerator(rescale=1./255.,validation_split=0.25)
print(traindf.genre)
train_generator=datagen.flow_from_dataframe(
dataframe=traindf,
directory="../data/images/",
x_col="_id",
y_col="genre",
has_ext=False,
batch_size=32, subset = 'training',
seed=42,
shuffle=True,
class_mode="categorical",
target_size=(32,32))
