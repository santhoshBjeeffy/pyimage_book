from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation
from keras.layers.core import Flatten,Dense
from keras import backend as k

class ShallowNet:
	@staticmethod

	def build(width,height,depth,classes):
		#initialize the model along with channel_last

		model=Sequential()
		input_shape=(height,width,depth)


		#update the image shape if "channel_first" being used

		if k.image_data_format() == "channels_first":

			input_shape=(depth,height,width)


		#define the first conv->Relu model

		model.add(Conv2D(32,(3,3),padding='same',input_shape=input_shape))
		model.add(Activation('relu'))

		#add a softmax classifier

		model.add(Flatten())
		model.add(Dense(classes))
		model.add(Activation('softmax'))


		return model