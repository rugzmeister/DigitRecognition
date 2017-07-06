from keras.models import Sequential
from keras.layers import Dense
from keras.datasets import mnist
from keras.models import model_from_json
from keras.utils import np_utils
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Conv2D, MaxPooling2D
import numpy as np

model=Sequential()
(xtrain,ytrain),(xtest,ytest)=mnist.load_data()
ytrain=np_utils.to_categorical(ytrain)
ytest=np_utils.to_categorical(ytest)
xtrain=xtrain.reshape(xtrain.shape[0],28,28,1)
xtest=xtest.reshape(xtest.shape[0],28,28,1)
xtrain=xtrain.astype('float32')
xtest=xtest.astype('float32')
xtrain/=255
xtest/=255
model.add(Conv2D(32,3,3,input_shape=(28,28,1),activation='relu'))
model.add(Conv2D(32,3,3,activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
#model.add(Dropout=0.25)
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dense(10,activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(xtrain,ytrain,epochs=10,batch_size=200)
model.save('model.h5')
print('model saved to disk!')

