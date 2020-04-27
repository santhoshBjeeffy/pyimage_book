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
from keras.models import load_model
import cv2

ap=argparse.ArgumentParser()

ap.add_argument("-d","--dataset",required=True,help="path to dataset")

ap.add_argument("-m","--model",required=True,help="path to save the model")
args=vars(ap.parse_args())



#initialize class labels 

classLabels=["cat","dog","panda"]
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

print(trainx.shape)
print(testx.shape)
print(trainy.shape)
print(testy.shape)
#convert the labels from integer to vectors

trainy=LabelBinarizer().fit_transform(trainy)
testy=LabelBinarizer().fit_transform(testy)


#initialize the model and optimizer

print("compiling the model")

opt=SGD(lr=0.001)
model=ShallowNet.build(width=32,height=32,depth=3,classes=3)
model.compile(loss="categorical_crossentropy",optimizer=opt,metrics=['accuracy'])

model.summary()
print("train the network")

H=model.fit(trainx,trainy,validation_data=(testx,testy),epochs=10,verbose=1)




#save the model

print("Saving the model...")
model.save(args["model"])
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


print("[INFO] loading pre-trained network...")
model = load_model(args["model"])

# make predictions on the images
print("[INFO] predicting...")
preds = model.predict(data, batch_size=2).argmax(axis=1)

#loop over sample images

for (i,imagepath) in enumerate(imagePaths):
	#load the example image predict it and display it

	image=cv2.imread(imagepath)
	cv2.putText(image, "Label: {}".format(classLabels[preds[i]]),
	(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
	cv2.imshow("Image", image)
	cv2.waitKey(0)