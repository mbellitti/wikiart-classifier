import numpy as np
import pandas as pd
from keras.models import Model
import matplotlib.pyplot as plt
from keras import optimizers
from keras.layers import Dense,GlobalAveragePooling2D
from keras.applications.vgg16 import VGG16
from keras_preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import time



def create_generators(nrows,img_size,feature="style",batch_size=32):
    """ Creates training and validation data generators.

        Input:
            nrows: int, how many rows of datframe or how many files
            img_size: (width,heigth), not all image sizes are allowed in vgg16
            feature: str, which column will be used as labels
            batch_size: int, minibatch size

        Output:
            nclass: int, # of classes to classify
            train_generator: a generator that would supply training image files of a certain batch_size to the model_fit module
            validation_generator: a generator that would supply validation image files of a certain batch_size to the model_fit module

        """


    # Split the data in train/test/validation
    df = pd.read_csv("../data/db.csv",nrows=nrows,na_values="?")
    df_train, df_test = train_test_split(df, test_size=0.2,shuffle=True)

    classes = set(df[feature])
    nclass = len(classes)

    train_datagen = ImageDataGenerator(
            featurewise_center=False,
            featurewise_std_normalization=False,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            rescale=1/255,
            validation_split=0.2)
            # TODO: need to think about featurewise normalization vs samplewise normalization


    train_generator = train_datagen.flow_from_dataframe(df_train,
                                    directory="../data/images/",
                                    x_col="_id",
                                    y_col=feature,
                                    has_ext=False,
                                    target_size=img_size,
                                    batch_size=batch_size,
                                    subset='training',
                                    classes=classes)


    validation_generator = train_datagen.flow_from_dataframe(df_train,
                                    directory="../data/images/",
                                    x_col="_id",
                                    y_col=feature,
                                    has_ext=False,
                                    target_size=img_size,
                                    batch_size=batch_size,
                                    subset='validation',
                                    classes=classes)

    test_datagen = ImageDataGenerator(
            featurewise_center=False,
            featurewise_std_normalization=False,
            rescale=1/255)
            # TODO: need to think about featurewise normalization vs samplewise normalization

    test_generator = test_datagen.flow_from_dataframe(df_test,
                                    directory="../data/images/",
                                    x_col="_id",
                                    y_col=feature,
                                    has_ext=False,
                                    target_size=img_size,
                                    batch_size=batch_size,
                                    classes=classes)

    return nclass, train_generator, validation_generator, test_generator


def cnn_layers_fn(nclass):
    """ CNN's top layer is pre-trained VGG16 followed by 3 dense layers and a softmax classifier

        Input:
            nclass: # of classes to classify

        Output:
           model: CNN
        """
    vgg_conv = VGG16(include_top=False, weights='imagenet', input_shape=(*img_size,3))

    #for i,layer in enumerate(vgg_conv.layers):
    #  print(i,layer.name)

    N_vgg16_layers=18 #number of layers there are in VGG16

    #adding dense layers after VGG16
    x = vgg_conv.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024,activation='relu')(x)
    x = Dense(1024,activation='relu')(x)
    x = Dense(512,activation='relu')(x)
    preds = Dense(nclass,activation='softmax')(x)
    model = Model(inputs=vgg_conv.input,outputs=preds)

    #layers taken from VGG16 should not be trained
    for layer in model.layers[:N_vgg16_layers]:
        layer.trainable=False
    for layer in model.layers[N_vgg16_layers:]:
        layer.trainable=True

    return model

if __name__ == '__main__':


    img_size = (48,48)#(32,32)#(224, 224)#
    batch_size = 32
    nrows = 1000
    epochs = 2

    # Loading data and create generators
    nclass, train_generator, validation_generator, test_generator = \
        create_generators(nrows,img_size,feature="style",batch_size=batch_size)

    # Create the CNN
    model = cnn_layers_fn(nclass)

    model.compile(optimizers.rmsprop(lr=0.0001, decay=1e-6),
                loss="categorical_crossentropy",metrics=["accuracy"])

    # Train
    t_in=time.time()

    history = model.fit_generator(
                    generator=train_generator,
                    validation_data=validation_generator,
                    epochs=epochs,
                    workers=4,
                    use_multiprocessing=True)

    t_end = time.time()
    t_run = int(t_end-t_in)

    d,h,m,s = t_run//86400, t_run//3600%24, t_run//60%60, t_run%60

    print("Finished training in {}d {}h {}m {}s".format(d,m,h,s))

    # Saving the history to see how well model is performing
    f=open('../data/loss_accuracy.dat', 'w')
    f.write("epochs=%d, running time={}d {}h {}m {}s \n".format(d,h,m,s))
    f.write('"train acc" \t \t "val acc"  \t \t "train loss" \t \t "val loss" \n')

    np.savetxt(f, np.transpose([ history.history['acc'],history.history['val_acc'], history.history['loss'], history.history['val_loss']]) , fmt='%.18f', delimiter='\t')
    f.close()

    # Test
    test_stat = model.evaluate_generator(generator=test_generator)
    test_results = dict(zip(model.metrics_names,test_stat))
