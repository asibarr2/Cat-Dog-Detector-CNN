import os
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split
import numpy as np
from os import listdir
import cv2
from skimage import io
#from scipy.misc import imresize
import keras
from keras.optimizers import Adam

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

img_size = 64
batch_size = 64
epochs = 50
#load image
data_path = '/home/rinzler/Desktop/DeepLearning_Spring2021/HW/HW3_Official_This_One/Train_Data'
labels = listdir(data_path)

x_cat = [];
x_dog = [];
cat_imgpath=listdir(data_path+'/'+labels[0])
dog_imgpath=listdir(data_path+'/'+labels[1])

for img in cat_imgpath:
   cat_img = io.imread(data_path+'/'+labels[0]+'/'+img)
   x_cat.append(cv2.resize(cat_img, (img_size, img_size)))

y_cat=np.ones(len(cat_imgpath))

for img in dog_imgpath:
   dog_img = io.imread(data_path + '/' + labels[1] + '/' + img)
   x_dog.append(cv2.resize(dog_img, (img_size, img_size)))

y_cat = np.zeros(len(cat_imgpath))
y_dog = np.ones(len(dog_imgpath))

x=np.asarray(x_cat+x_dog)
y=np.append(y_cat,y_dog)

y = keras.utils.to_categorical(y)

train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.1, random_state=30)

model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(img_size, img_size, 3)))
model.add(Activation('sigmoid'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
# model.add(Dense(64))
# model.add(Dropout(0.1))
# model.add(Activation('sigmoid'))
model.add(Dense(2))
model.add(Activation('softmax'))
adamop=Adam(lr=0.1)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(train_x, train_y,
          batch_size=batch_size,
          epochs=epochs,
          shuffle=True)

score = model.evaluate(test_x, test_y)
pred=model.predict(test_x)

print('Test loss:', score[0])
print('Test accuracy:', score[1])

# get output of a layer from keras model
