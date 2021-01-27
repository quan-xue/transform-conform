import numpy as np
import os
import argparse
import cv2
from imutils import paths
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Layer,Conv2D,Activation,MaxPool2D,Dense,Flatten,Dropout
import tensorflow as tf
from tensorflow import keras

class SmileDetector():
	
	def __init__(self, inference_path, face_cascade="haarcascade_frontalface_default.xml", model_path="smile.hdf5"):
		self.face_cascade = cv2.CascadeClassifier(face_cascade)
		self.model_path = model_path
		self.inference_path = inference_path
		self.input_shape = (32,32,1)

	def create_model(self):
		model = Sequential()
		
		model.add(Conv2D(32, (3, 3), input_shape=self.input_shape))
		model.add(Activation('relu'))
		model.add(MaxPool2D(pool_size=(2, 2)))

		model.add(Conv2D(64, (3, 3)))
		model.add(Activation('relu'))
		model.add(MaxPool2D(pool_size=(2, 2)))

		model.add(Conv2D(128, (3, 3)))
		model.add(Activation('relu'))
		model.add(MaxPool2D(pool_size=(2, 2)))

		model.add(Flatten())
		model.add(Dense(256))
		model.add(Activation('relu'))
		model.add(Dropout(0.2))
		model.add(Dense(128))
		model.add(Dropout(0.2))
		model.add(Activation('relu'))
		model.add(Dense(1))
		model.add(Activation('sigmoid'))

		model.compile(loss='binary_crossentropy',
					optimizer='adam',
					metrics=['accuracy'])
		
		model.summary()
		
		return model

	def process_one_image(self, image_name):
		height, width = 32, 32

		image = cv2.imread(os.path.join(self.inference_path, image_name))
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		faces = self.face_cascade.detectMultiScale(gray, 1.1, 3, minSize=(100, 100))
		for (x,y,w,h) in faces:
			face_clip = gray[y:y+h, x:x+w]
		
		resize_face_clip = cv2.resize(face_clip,(height,width),interpolation=cv2.INTER_AREA)
		processed_image = img_to_array(resize_face_clip)/255.0 # normalise
		
		return processed_image

	def detect_smile(self, image_name):
		
		# Create a new model instance
		model = self.create_model()

		# Restore the weights
		model.load_weights(self.model_path)

		processed_image = self.process_one_image(image_name)

		processed_image = processed_image.reshape(1,32,32,1)

		prediction = model.predict(processed_image)
		print(prediction)
		return 1 if prediction >= 0.5 else 0


test = SmileDetector(inference_path="image")
test.detect_smile("test3.jpg")