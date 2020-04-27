from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
from sklearn import datasets
import matplotlib.pyplot as plt 
import numpy as np 
import argparse

#construct the argument parser and parse the arguments
ap=argparse.ArgumentParser()

ap.add_argument('-o','--output',required=True,help='path to the output loss/accuracy plot')
args=vars(ap.parse_args())

#get the mnist data

print("Info loading mnist data set")
dataset=datasets.fetch_mldata("MNIST original")


#scale the raw pixel instensities to the range [0,1] then construct training and testing split

data=dataset.data.astype('float')/255.0

(train_x,test_x,train_y,test_y)=train_test_split(data,dataset.target,test_Size=0.33)

print(train_x.shape)
print(train_y.shape)

#convert the label from integer to vector

lb=LabelBinarizer()

train_y=lb.fit_transform(train_y)
test_y=lb.fit_transform(test_y)

print(train_y)

#define architectur

model=Sequential()
model.add(Dense(256,input_shape=data[1],activation='sigmoid'))
model.add(Dense(128,activation='sigmoid'))
model.add(Dense(64,activation='sigmoid'))
model.add(Dense(10,activation='softmax'))


#train model using SGD
model.compile(loss='categorical_crossentropy',optimizer='SGD',metrics=['accuracy'])

H=model.fit(train_x,train_y,validation_data=(test_x,test_y),epochs=20,batch_size=128)

#test the network

print("Testing the model")

predictions=model.predict(test_x,batch_size=128)
print(classification_report(test_y.argmax(axis=1), predictions.argmax(axis=1),
                            target_names=[str(x) for x in lb.classes_]))

plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(0, 100), H.history['loss'], label='train_loss')
plt.plot(np.arange(0, 100), H.history['val_loss'], label='val_loss')
plt.plot(np.arange(0, 100), H.history['acc'], label='train_acc')
plt.plot(np.arange(0, 100), H.history['val_acc'], label='val_acc')
plt.title('Training Loss & Accuracy')
plt.xlabel('Epoch #')
plt.ylabel('Loss/Accuracy')
plt.legend()
plt.savefig(args['output'])