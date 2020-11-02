import skimage
from skimage import io,color,transform
import matplotlib
from matplotlib import pyplot as plt
import os
import numpy as np
import math
import tensorflow as tf
import json
import random
import cv2


import keras
from keras.layers.advanced_activations import LeakyReLU
import keras.backend as K
from keras.models import Sequential,load_model
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense

dirr = r'C:\Users\Admin\Desktop\Project YOLO\COCO\\'

with open(r'C:\Users\Admin\Desktop\Project YOLO\data.json') as f:
    data = json.load(f)

Data = data[:2000]

def seed():
    return 0.6

#random.shuffle(Data,seed)
x = []
y = []
for i in Data[:100]:
    img = io.imread(dirr+i['name'])
    x.append(img/255)
    y.append(i['grid'])
X = np.array(x)
Y = np.array(y)




def loss1(labels,preds):
    xloss = 10*K.sum(K.pow((labels[:,:,:,1]-preds[:,:,:,1])*labels[:,:,:,0],2))
    yloss = 10*K.sum(K.pow((labels[:,:,:,2]-preds[:,:,:,2])*labels[:,:,:,0],2))
    wloss = 10*K.sum(K.pow((labels[:,:,:,3]-preds[:,:,:,3])*labels[:,:,:,0],2))
    hloss = 10*K.sum(K.pow((labels[:,:,:,4]-preds[:,:,:,4])*labels[:,:,:,0],2))
    closs1 = 2*K.sum(K.pow((labels[:,:,:,0]-preds[:,:,:,0])*labels[:,:,:,0],2))
    closs2 = K.sum(K.pow((labels[:,:,:,0]-preds[:,:,:,0]),2))
    
        
                
                
    total_loss = xloss+yloss+wloss+hloss+closs1+closs2
    return total_loss


    
model = Sequential()

#Model
model = Sequential()
model.add(Conv2D(16,(3,3),input_shape=(448,448,3),padding = 'same'))#1
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32,(3,3),padding='same'))#2
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3),padding='same'))#3
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(128,(3,3),padding='same'))#4
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(256,(3,3),padding='same'))#5
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(512,(3,3),padding='same'))#6
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(1024,(3,3),padding='same'))#7
model.add(LeakyReLU(alpha=0.1))

model.add(Conv2D(1024,(3,3),padding='same'))#8
model.add(LeakyReLU(alpha=0.1))

model.add(Conv2D(5,(3,3),activation='sigmoid',padding='same'))





opt = keras.optimizers.SGD(learning_rate=0.001,momentum=0.2)
model.compile(loss=loss1,optimizer=opt, metrics=['accuracy'])


#model1 = load_model('model4.h5',custom_objects={ 'loss1': loss1})
model.load_weights(r'model1x.h5')
#model.fit(X,Y,epochs=5,batch_size=1)

#model.save('model1x.h5')
prd = model.predict(X[:10])

img = X[1]
y = prd[0,3,3,1]*448
x = prd[0,3,3,2]*448
h = prd[0,3,3,4]
h = h*h*448
w = prd[0,3,3,3]
w = w*w*448


cv2.rectangle(img,(int(x-w/2),int(y-h/2)),(int(x+w/2),int(y+h/2)),[1,0,0],5)
plt.imshow(img)
plt.show()




    


