from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense,Dropout,Activation,Flatten,BatchNormalization
from keras.layers import Conv2D,MaxPooling2D
import os
import keras

num_classes = 7
img_row_size,img_column_size = 48,48
batch_size = 32

train_data_dir = 'images/train'
validation_data_dir = 'images/validation'

train_datagen = ImageDataGenerator(
					rescale=1./255,
					rotation_range=30,
					shear_range=0.3,
					zoom_range=0.3,
					width_shift_range=0.4,
					height_shift_range=0.4,
					horizontal_flip=True,
					fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
					train_data_dir,
					color_mode='grayscale',
					target_size=(img_row_size,img_column_size),
					batch_size=batch_size,
					class_mode='categorical',
					shuffle=True)

validation_generator = validation_datagen.flow_from_directory(
							validation_data_dir,
							color_mode='grayscale',
							target_size=(img_row_size,img_column_size),
							batch_size=batch_size,
							class_mode='categorical',
							shuffle=True)




#Create Model
model = keras.Sequential()
model.add(Conv2D(64, (3,3), activation='relu', input_shape=(img_row_size,img_column_size,1)))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(Conv2D(64, (3,3), activation='relu'))
#model.add(Dropout(0.1))

model.add(Conv2D(128, (3,3), activation='relu'))
model.add(Conv2D(128, (3,3), activation='relu'))
model.add(MaxPooling2D(2,2))
#model.add(Dropout(0.1))

model.add(Conv2D(128, (3,3), activation='relu'))
model.add(Conv2D(128, (3,3), activation='relu'))
#model.add(Dropout(0.1))

model.add(Conv2D(256, (3,3), activation='relu'))
model.add(Conv2D(256, (3,3), activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Flatten())
#model.add(Dropout(0.1))



model.add(Dense(256,activation='relu'))
#model.add(Dropout(0.1))

model.add(Dense(128,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
print(model.summary())

from keras.optimizers import Adam,RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

adam = Adam(lr=0.001)
checkpoint = ModelCheckpoint('face_condition.h5',
                             monitor='val_loss',
                             mode='min',
                             save_best_only=True,
                             verbose=1)

earlystop = EarlyStopping(monitor='val_loss',
                          min_delta=0,
                          patience=4,
                          verbose=1,
                          restore_best_weights=True
                          )

reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                              factor=0.2,
                              patience=3,
                              verbose=1,
                              min_delta=0.0001)

callbacks = [earlystop,checkpoint]

model.compile(loss='categorical_crossentropy',
              optimizer = RMSprop(lr=0.0001),
              metrics=['accuracy'])

nb_train_samples = 28821
nb_validation_samples = 7066
epochs=50

history=model.fit_generator(
                train_generator,
                steps_per_epoch=nb_train_samples//batch_size,
                epochs=epochs,
                callbacks=callbacks,
                validation_data=validation_generator,
                validation_steps=nb_validation_samples//batch_size)























































