
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from santhob.preprocessing import ImageToArrayPreProcessor
from santhob.preprocessing import SimplePreprocessor
from santhob.datasets import SimpleDatasetLoader
from santhob.nn.cnn import ShallowNet
from keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt 
import numpy as np 
import argparse
from tqdm import tqdm
import os

DATADIR = "/home/guru/Desktop/pyimagebook/chapter12/santhob/datasets"

CATEGORIES = ["dog", "cat","panda"]

for category in CATEGORIES:  # do dogs and cats
    path = os.path.join(DATADIR,category)  # create path to dogs and cats
    for img in os.listdir(path):  # iterate over each image per dogs and cats
        img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array
        plt.imshow(img_array, cmap='gray')  # graph it
        plt.show()  # display!

        break  # we just want one for now so break
    break 

#grab the list of images that we are describing

print("Images are loading...")
imagePaths = list(paths.list_images(args["dataset"]))
#imagepaths = list(paths.list_images(args['dataset']))

#initialize the image preprocessor

sp=SimplePreprocessor(32,32)
iap=ImageToArrayPreProcessor()

#load the dataset from disk and scale to range between [0,1]

sd1=SimpleDatasetLoader(preprocessors=[sp,iap])
(data,labels)=sd1.load(imagePaths,verbose=500)

data=data.astype("float") / 255.0

#partition the dataser to train and test dataset

(trainx,testx,trainy,testy)=train_test_split(data,labels,test_size=0.25,random_state=42)

#convert the labels from integer to vectors

trainy=LabelBinarizer().fit_transform(trainy)
testy=LabelBinarizer().fit_transform(testy)


#initialize the model and optimizer

print("compiling the model")

opt=SGD(lr=0.001)
model=ShallowNet.build(width=32,height=32,depth=3,classes=3)
model.compile(loss="categorical_crossentropy",optimizer=opt,metrics=['accuracy'])

print("train the network")

H=model.fit(trainx,trainy,validation_data=(testx,testy),epochs=10,verbose=1)

#evaluate the model

print("evaluating the model....")
predictions=model.predict(testx)
print(classification_report(testy.argmax(axis=1),predictions.argmax(axis=1),target_names=["cat","dog","panda"]))


# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 10), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 10), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 10), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, 10), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()