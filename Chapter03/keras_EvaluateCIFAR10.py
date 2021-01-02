# Bagged


import numpy as np
import scipy.misc
from keras.models import model_from_json
from keras.optimizers import SGD
import cv2

#load model
model_architecture = 'cifar10_architecture.json'
model_weights = 'cifar10_weights.h5'
model = model_from_json(open(model_architecture).read())
model.load_weights(model_weights)

#load images
img_names = ['../Datasets/cat.jpg']
#image = cv2.imread(imagePath)
#image = cv2.resize(image, (128, 128)).flatten()

imgs = [np.transpose(cv2.resize(cv2.imread(img_name), (32, 32)),
                     (1, 0, 2)).astype('float32')
           for img_name in img_names]
imgs = np.array(imgs) / 255

# train
optim = SGD()
model.compile(loss='categorical_crossentropy', optimizer=optim,
	metrics=['accuracy'])
 
predictions = model.predict_classes(imgs)
print(predictions)

