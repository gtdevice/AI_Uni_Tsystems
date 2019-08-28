import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import cv2
import os
import random
from PIL import Image
 # Keras Libraries
import keras
from keras.models import Sequential
from keras.layers import Conv2D, Dropout
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator, load_img

total_images_train_normal = os.listdir('xray/chest_xray/train/NORMAL/')
total_images_train_pneumonia = os.listdir('xray/chest_xray/train/PNEUMONIA/')
data_dir = Path('/xray/chest_xray')
# Path to train directory (Fancy pathlib...no more os.path!!)
train_dir = data_dir / 'train'
# Path to validation directory
val_dir = data_dir / 'val'
# Path to test directory
test_dir = data_dir / 'test'
sample_normal = random.sample(total_images_train_normal,6)
f,ax = plt.subplots(2,3,figsize=(15,9))

for i in range(0,6):
    im = cv2.imread('xray/chest_xray/train/NORMAL/'+sample_normal[i])
    ax[i//3,i%3].imshow(im)
    ax[i//3,i%3].axis('off')
f.suptitle('Normal Lungs')
plt.show()
sample_pneumonia = random.sample(total_images_train_pneumonia,6)
f,ax = plt.subplots(2,3,figsize=(15,9))

for i in range(0,6):
    im = cv2.imread('xray/chest_xray/train/PNEUMONIA/'+sample_pneumonia[i])
    ax[i//3,i%3].imshow(im)
    ax[i//3,i%3].axis('off')
f.suptitle('Pneumonia Lungs')
plt.show()
#total_images_train_pneumonia
f=plt.figure(1)
h=[]
w=[]
c=[]
for i in total_images_train_pneumonia:
    img = cv2.imread('xray/chest_xray/train/PNEUMONIA/'+i)
    #cv2.imshow("Image", img)
    height, width, channels = img.shape
    h.append(height)
    w.append(width)
    c.append(channels)
f.suptitle('Image sizes')
f,ax = plt.subplots(1,3,figsize=(15,9))
ax[0].plot(h)
ax[0].suptitle('height')
ax[1].plot(w)
ax[1].suptitle('width')
ax[2].plot(c)
ax[2].suptitle('channels')
plt.show()



#CNN model
cnn = Sequential()
# 1st Convolution
cnn.add(Conv2D(32, (3, 3), activation="relu", input_shape=(200, 200, 3)))
# 1st Pooling
cnn.add(MaxPooling2D(pool_size = (2, 2)))
# 2nd Convolution
cnn.add(Conv2D(64, (3, 3), activation="relu"))
# 2nd Pooling layer
cnn.add(MaxPooling2D(pool_size = (2, 2)))
# 3nd Convolution
cnn.add(Conv2D(128, (3, 3), activation="relu"))
# 3nd Pooling layer
cnn.add(MaxPooling2D(pool_size = (2, 2)))

# 3nd Convolution
cnn.add(Conv2D(128, (3, 3), activation="relu"))

# 3nd Pooling layer
cnn.add(MaxPooling2D(pool_size = (2, 2)))

# Flatten the layer
cnn.add(Flatten())

cnn.add(Dropout(0.5, input_shape=(60,)))

# Fully Connected Layers
cnn.add(Dense(activation = 'relu', units = 512))
cnn.add(Dense(activation = 'sigmoid', units = 1))

# Compile the Neural network
cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# Fitting the CNN to the images
# The function ImageDataGenerator augments your image by iterating through image as your CNN is getting ready to process that image

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)  #Image normalization.

training_set = train_datagen.flow_from_directory('xray/chest_xray/train',
                                                 target_size = (200, 200),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

validation_generator = test_datagen.flow_from_directory('xray/chest_xray/val/',
    target_size=(200, 200),
    batch_size=32,
    class_mode='binary')

test_set = test_datagen.flow_from_directory('xray/chest_xray/test',
                                            target_size = (200, 200),
                                            batch_size = 32,
                                            class_mode = 'binary')

cnn.summary()

cnn_model = cnn.fit_generator(training_set,
                         steps_per_epoch = 30,
                         epochs = 10,
                         validation_data = validation_generator,
                         validation_steps = 200)

test_accu = cnn.evaluate_generator(test_set,steps=200)
print('The testing accuracy is :',test_accu[1]*100, '%')
plt.plot(cnn_model.history['acc'])
plt.plot(cnn_model.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training set', 'Validation set'], loc='upper left')
plt.show()

plt.plot(cnn_model.history['val_loss'])
plt.plot(cnn_model.history['loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training set', 'Test set'], loc='upper left')
plt.show()