
# coding: utf-8

# In[9]:

import keras
import csv
import cv2
from keras.models import Sequential
from keras.regularizers import l2, activity_l2
from keras.layers import Dense,Input,Flatten,Lambda,Activation,Conv2D, MaxPooling2D,Cropping2D,Dropout,Reshape,SpatialDropout2D
import numpy as np
import h5py
from keras import backend as K


# Pre-processing the functions

def read_driver_file(location):
    with open(location) as file:
        reader = csv.reader(file)
        lines =[line for line in reader]    
    return lines

# Read an image from a file
def read_image(image_loc):
    return cv2.imread(image_loc)

# Pre-processing: change color of an image to YUV
def pre_process(image):
    new_img = cv2.GaussianBlur(image, (3,3), 0)
    new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2YUV)
    return new_img

# Creating a training data sets
def create_training_data(driver_loc,image_base_loc):
    lines = read_driver_file(driver_loc)
    images = []
    measurements = []
    correction_fact = 0.25
    for line in lines:
        image_name_c = line[0].split('/')[-1]
        image_name_l = line[1].split('/')[-1]
        image_name_r = line[2].split('/')[-1]
        
        image_location_c = image_base_loc+'/IMG/'+image_name_c
        image_location_l = image_base_loc+'/IMG/'+image_name_l
        image_location_r = image_base_loc+'/IMG/'+image_name_r
        image_c = read_image(image_location_c)
        image_l = read_image(image_location_l)
        image_r = read_image(image_location_r)
        images.append(pre_process(image_c))
        images.append(pre_process(image_l))
        images.append(pre_process(image_r))
        m = float(line[3])
        measurements.append(m)
        measurements.append(m + correction_fact)
        measurements.append(m - correction_fact)
    return (images,measurements)

# flip a image
def augmentation(images, measurements):
    images_augmentation = []
    measurements_augmentation = []
    for image,measurement in zip(images, measurements):
        images_augmentation.append(image)
        measurements_augmentation.append(measurement)
        images_augmentation.append(cv2.flip(image,1))
        measurements_augmentation.append(measurement*-1.0)
    return (images_augmentation,measurements_augmentation)

'''calling function to get training and label data set'''
images, measurements = create_training_data('./data/driving_log.csv','./data')
images_augmentation,measurements_augmentation = augmentation(images, measurements)

# converting to numpy array
X_train = np.array(images_augmentation)
y_train = np.array(measurements_augmentation)



'''defining a architecture' and creating a model'''
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
fig = plt.figure()
p = X_train[0]
fig.add_subplot(1,1,1)
plt.imshow(p)
# fig.add_subplot(2,1,2)
# plt.imshow(p)
plt.show()


''' Architecture of model inspired by Invidia'''
shape = X_train[0].shape
# from keras.preprocessing.image import ImageDataGenerator
def get_model():
    model = Sequential()
#     model.add(Lambda(lambda x:tfk.image.resize_images(x,(128,128)),input_shape=shape))
    #model.add(Reshape((128,128)))
    model.add(Lambda(lambda x: x/255.0 - 0.5,input_shape=shape))
    model.add(Cropping2D(cropping=((80,20),(0,0))))
    
    model.add(Conv2D(24,5,5, subsample=(2,2), border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
#     model.add(SpatialDropout2D(0.2))
    
    model.add(Conv2D(36,5,5, subsample=(2,2),border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
#     model.add(SpatialDropout2D(0.2))
    
    model.add(Conv2D(48,5,5, subsample=(2,2),border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
    
    model.add(Conv2D(64,3,3,activation='relu',border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
    model.add(Conv2D(64,3,3,activation='relu',border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

    model.add(Flatten())

    model.add(Dense(1164))
    model.add(Activation('relu'))

    model.add(Dense(100))
    model.add(Activation('relu'))

    model.add(Dense(50))
    model.add(Activation('relu'))

    model.add(Dense(10))
    model.add(Activation('relu'))


    model.add(Dense(1))
    return model

'''training a model and defining loss and optimization function'''
model = get_model()
model.compile(loss='mse', optimizer='adam')
model.fit(X_train,y_train,validation_split=0.2,shuffle=True,nb_epoch=5,batch_size=128)


model.summary()
model.save('model.h5')

