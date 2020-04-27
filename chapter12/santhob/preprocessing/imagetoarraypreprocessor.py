from keras.preprocessing.image import img_to_array

class ImageToArrayPreProcessor:

	def __init__(self,data_format=None):

		#store the image data format

		self.data_format=data_format


	def preprocess(self,image):

		#apply keras utility function to rearranges the dimensions of image

		return img_to_array(image,data_format=self.data_format)

