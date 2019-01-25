

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization
chanDim = -1
model = Sequential()
model.add(Conv2D(filters = 128, kernel_size =  (3,3), input_shape = (64,64,1), padding= 'same'))
model.add(BatchNormalization(axis=chanDim))
model.add(Activation("elu"))

model.add(Conv2D(filters = 128, kernel_size =  (3,3), padding= 'same'))
model.add(BatchNormalization(axis=chanDim))
model.add(Activation("elu"))
#model.add(BatchNormalization(axis=chanDim))
model.add(Conv2D(filters = 128, kernel_size =  (3,3), padding= 'same'))
model.add(BatchNormalization(axis=chanDim))
model.add(Activation("elu"))
#model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size = (2,2),strides=(2, 2)))

model.add(Conv2D(filters = 128, kernel_size =  (3,3), padding = 'same'))
model.add(BatchNormalization(axis=chanDim))
model.add(Activation("elu"))
#model.add(BatchNormalization(axis=chanDim))
model.add(Conv2D(filters = 128, kernel_size =  (3,3), padding= 'same'))
model.add(BatchNormalization(axis=chanDim))
model.add(Activation("elu"))
#model.add(BatchNormalization(axis=chanDim))
model.add(Conv2D(filters = 128, kernel_size =  (3,3), padding= 'same'))
model.add(BatchNormalization(axis=chanDim))
model.add(Activation("elu"))
#model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size = (2,2),strides=(2, 2)))

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding= 'same'))
model.add(BatchNormalization(axis=chanDim))
model.add(Activation("elu"))
#model.add(BatchNormalization(axis=chanDim))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding= 'same'))
model.add(BatchNormalization(axis=chanDim))
model.add(Activation("elu"))
#model.add(BatchNormalization(axis=chanDim))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding= 'same'))
model.add(BatchNormalization(axis=chanDim))
model.add(Activation("elu"))
#model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size = (2,2), strides=(2, 2)))


model.add(Conv2D(filters = 64, kernel_size = (3,3),padding= 'same'))
model.add(BatchNormalization(axis=chanDim))
model.add(Activation("elu"))
#model.add(BatchNormalization(axis=chanDim))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding= 'same'))
model.add(BatchNormalization(axis=chanDim))
model.add(Activation("elu"))
#model.add(BatchNormalization(axis=chanDim))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding= 'same'))
model.add(BatchNormalization(axis=chanDim))
model.add(Activation("elu"))
#model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size = (2,2), strides=(2, 2)))
model.add(Flatten())
model.add(Dense(256))
model.add(Activation("elu"))
model.add(Dropout(0.25))
model.add(Dense(32))
model.add(Activation("elu"))
model.add(Dropout(0.25))
model.add(Dense(2))
model.add(Activation("softmax"))
model.summary()
import os
fname = "smile_weights-{epoch:03d}-{val_loss:.4f}.hdf5"
from keras.callbacks import ModelCheckpoint
checkpoint = ModelCheckpoint(fname, monitor="val_loss",
save_best_only=True, verbose=1)
callbacks = [checkpoint]
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1/255,
				shear_range = 0.2,	
				zoom_range = 0.2,
    				rotation_range=30,
    				width_shift_range=0.2,
   				height_shift_range=0.2,
				fill_mode = 'nearest')
train_generator = train_datagen.flow_from_directory(
        '/home/sait/DL4CVStarterBundle/datasets/SMILEsmileD/train',
        target_size=(64, 64),
        batch_size=8,
	shuffle=True,
        color_mode = 'grayscale',
        class_mode='categorical',
        save_format='jpeg')
dev_datagen = ImageDataGenerator(rescale=1/255)
validation_generator = dev_datagen.flow_from_directory(
                '/home/sait/DL4CVStarterBundle/datasets/SMILEsmileD/validation',
        target_size=(64, 64),
        batch_size=8,
        color_mode ='grayscale',
	shuffle=True,
        class_mode='categorical')
test_datagen =  ImageDataGenerator(rescale = 1/255)
test_generator = test_datagen.flow_from_directory(
	'/home/sait/DL4CVStarterBundle/datasets/SMILEsmileD/test',
	target_size = (64,64),
	batch_size = 8,
        color_mode = 'grayscale',
	shuffle=True,
	class_mode = 'categorical')
from keras.optimizers import SGD

optimizer =  SGD(0.0001, decay= 0.01/40, momentum = 0.9, nesterov = 'true')
model.compile( optimizer = optimizer , loss='categorical_crossentropy', metrics=['accuracy'])


H = model.fit_generator(
        train_generator,
	callbacks = callbacks,
        steps_per_epoch=1494,
        epochs=20,
        verbose = 1,
        validation_data=validation_generator,
        validation_steps=88)

import numpy as np 
import matplotlib.pyplot as plt 

plt.style.use("ggplot")
fig = plt.figure()


plt.plot(np.arange(0, 20), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 20), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 20), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, 20), H.history["val_acc"], label="val_acc")
title = "0.01"
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch#")

plt.ylabel("Loss/Accuracy")
plt.legend()
fig.savefig('image.png', dpi=fig.dpi)
plt.show()

