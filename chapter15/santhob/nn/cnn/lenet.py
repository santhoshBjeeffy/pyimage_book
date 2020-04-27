from keras.models import Sequential
from keras.layers.convolutional import Conv2D,MaxPooling2D
from keras.layers.core import Activation,Flatten,Dense
from keras import backend as k 

class LeNet:
	@staticmethod

	def build(width,height,depth,classes):

		model=Sequential()
		input_shape=(height,width,depth)


		if k.image_data_format()=="channels_first":
			input_shape=(depth,height,width)



		#first set of conv->RElu->Pool

		model.add(Conv2D(20,(5,5),padding='same',input_shape=input_shape))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2,2),stride=(2,2)))


		#second set of conv->relu->pool

		model.add(Conv2D(50,(5,5),padding='same'))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2,2),stride=(2,2)))


		#first and only FC=>Relu layers

		model.add(Flatten())
		model.add(Dense(500))
		model.add(Activation("relu"))



		#Softmax final layer

		model.add(Dense(classes))
		model.add(Activation("Softmax"))


		return model







