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



def train_validation_gen_fn(nrows, img_size,batch_size=32):
    """ Takes the dataframe and the path to a directory and generates batches of augmented/normalized data.

        Input:
            nrows=how many rows of datframe or how many files
            img_size= (length,breadth); not all image sizes are allowed in vgg16
            batch_size=how many files to include in a single batch
        
        Output:
            nclass=# of classes to classify 
            train_generator=a generator that would supply training image files of a certain batch_size to the model_fit module
            validation_generator=a generator that would supply validation image files of a certain batch_size to the model_fit module
        
        """

    df = pd.read_csv("../data/db.csv",nrows=nrows,na_values="?")
    df_train, df_test = train_test_split(df, test_size=0.1,shuffle=True)
    feature = "style"
    classes=set(df[feature])
    nclass = len(classes)
    print("nclass", nclass)
    datagen = ImageDataGenerator(featurewise_center=False,
        featurewise_std_normalization=False,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,rescale=1/255,validation_split=0.2)
    #need to think about featurewise normalization vs saplewise normalization

    
   
    train_generator = datagen.flow_from_dataframe(df_train,directory="../data/images/",
                                    x_col="_id",
                                    has_ext=False,
                                    target_size=img_size,
                                    y_col=feature, 
                                    batch_size=batch_size,
                                    subset='training',
                                    classes = classes)


    validation_generator = datagen.flow_from_dataframe(df_train,directory="../data/images/",
                                    x_col="_id",
                                    has_ext=False,
                                    target_size=img_size,
                                    y_col=feature, 
                                    batch_size=batch_size,
                                    subset='validation',
                                    classes = classes)


    return nclass, train_generator,validation_generator


def CNN_layers_fn(nclass): 
    """ CNN's top layer is pre-trained VGG16 followed by 3 dense layers and a softmax classifier

        Input:
            nclass=# of classes to classify
        
        Output:
           model= CNN   
        """
    vgg_conv=VGG16(include_top=False, weights='imagenet', input_shape=(*img_size,3))

    #for i,layer in enumerate(vgg_conv.layers):
    #  print(i,layer.name)

    N_vgg16_layers=18 #number of layers there are in VGG16


    #adding dense layers after VGG16
    x=vgg_conv.output
    x=GlobalAveragePooling2D()(x)
    x=Dense(1024,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
    x=Dense(1024,activation='relu')(x) #dense layer 2
    x=Dense(512,activation='relu')(x) #dense layer 3
    preds=Dense(nclass,activation='softmax')(x) #final layer with softmax activation
    model=Model(inputs=vgg_conv.input,outputs=preds)
    
    #layers taken from VGG16 should not trained.
    for layer in model.layers[:N_vgg16_layers]:
        layer.trainable=False
    for layer in model.layers[N_vgg16_layers:]:
        layer.trainable=True

    return model

def train_model_fn(model, train_generator, validation_generator,epochs):
    """ Takes the dataframe and the path to a directory and generates batches of augmented/normalized data.

        Input:
            train_generator=a generator that would supply training image files of a certain batch_size to the model_fit module
            validation_generator=a generator that would supply validation image files of a certain batch_size to the model_fit module
            epochs=how many times optimizer should run over all mini-batches 

        Output:
           history= it has a list of losses and accuracy over training and validation data
                history.history['acc']
                history.history['val_acc']
                history.history['loss']
                history.history['val_loss']
                            
    """

    ##choosing optimizers    

    model.compile(optimizers.rmsprop(lr=0.0001, decay=1e-6),
                loss="categorical_crossentropy",metrics=["accuracy"])


    history=model.fit_generator(generator=train_generator, validation_data = validation_generator,
                    epochs=epochs)

    return history
    




t_in=time.time()


img_size=(48,48)#(32,32)#(224, 224)#
batch_size=32
nrows =1000
epochs=10

#loading the data
nclass, train_generator,validation_generator=train_validation_gen_fn(nrows, img_size,batch_size)

#creating the CNN model
model=CNN_layers_fn(nclass)

#fitting the model to data
history=train_model_fn(model, train_generator, validation_generator,epochs)

t_end=time.time()
t_code=(t_end-t_in)/60

#saving the history to see how well model is predicting over validation set
f=open('../data/loss_accuracy.dat', 'w')
f.write("epochs=%d, code time=%f (in min) \n"  %(epochs,t_code))
f.write('"train acc" \t \t "val acc"  \t \t "train loss" \t \t "val loss" \n')

np.savetxt(f, np.transpose([ history.history['acc'],history.history['val_acc'], history.history['loss'], history.history['val_loss']]) , fmt='%.18f', delimiter='\t')
f.close()


#model.fit_generator(generator=train_generator,  epochs=2)

#test_generator = datagen.flow_from_dataframe(df_test,
#                                directory="../data/images/",
#                                x_col="_id",has_ext=False, target_size=img_size,
#                                y_col=feature,batch_size=batch_size, classes = classes)

#test_stat=model.evaluate_generator(generator=test_generator)
#print(test_stat)

"""
Following is the code that can be used to see wrongly-classified images
"""

"""
fnames = test_generator.filenames
 
ground_truth = test_generator.classes
 
label2index = test_generator.class_indices
 
# Getting the mapping from class index to class label
idx2label = dict((v,k) for k,v in label2index.items())
 
predictions = model.predict_classes(test_features)
prob = model.predict(test_features)
 
errors = np.where(predictions != ground_truth)[0]
print("No of errors = {}/{}".format(len(errors),nTest))


print(predictions[errors[0]],ground_truth[errors[0]] )

for i in range(10):#len(errors)):
    pred_class_index = np.argmax(prob[errors[i]]) #predicted class is the one corresponding to which CNN gives the maximum probability 
    pred_class_label = idx2label[pred_class_index]
    actual_class_index=ground_truth[errors[i]]
    acutal_class_label=idx2label[actual_class_index] 
    print('Original label:{}, Prediction :{}, confidence : {:.3f}'.format(
        acutal_class_label,
        pred_class_label,
        prob[errors[i]][pred_class_index]))
    original = load_img('{}/{}'.format("../data/images/",fnames[errors[i]]))
    plt.imshow(original)
    plt.show()

"""