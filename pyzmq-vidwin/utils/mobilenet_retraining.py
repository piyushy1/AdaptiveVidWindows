# Author - Piyush Yadav
# Insight Centre for Data Analytics
# Package- VidWIN Project

import tensorflow.keras as k
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
#from keras.models import Model
from tensorflow.keras.models import Model
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.layers import Dense,GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input
import numpy as np
from time import time
from tensorflow.keras.callbacks import TensorBoard
#from IPython.display import Image


#mobile = k.applications.mobilenet.MobileNet()

# function to upload the image
def prepare_image(file):
    img_path = ''
    img = image.load_img(img_path + file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return k.applications.mobilenet.preprocess_input(img_array_expanded_dims)

# preprocessed_image = prepare_image('/home/dhaval/piyush/Usecases_dataset/pascal voc dataaset/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/JPEGImages/2007_001288.jpg')
# predictions = mobile.predict(preprocessed_image)
# results = imagenet_utils.decode_predictions(predictions)
# print(results)


base_model=MobileNet(weights='imagenet',include_top=False) #imports the mobilenet model

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x=Dense(1024,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
x=Dense(1024,activation='relu')(x) #dense layer 2
x=Dense(512,activation='relu')(x) #dense layer 3
# and a logistic layer -- let's say we have 20 voc classes
preds = Dense(20, activation='softmax')(x)

model=Model(inputs=base_model.input,outputs=preds) ##now a model has been created based on our architecture

# for i,layer in enumerate(model.layers):
#     print(i, layer.name)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# the first 249 layers and unfreeze the rest:
# for layer in model.layers[:20]:
#    layer.trainable = False
# for layer in model.layers[20:]:
#    layer.trainable = True
model.compile(optimizer= SGD(learning_rate=0.01, momentum=0.0, nesterov=False, name='SGD'),loss='categorical_crossentropy',metrics=['accuracy'])
#model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])
# Adam optimizer
# loss function will be categorical cross entropy
# evaluation metric will be accuracy

# call the dataset

train_datagen=ImageDataGenerator(preprocessing_function=preprocess_input) #included in our dependencies

train_generator=train_datagen.flow_from_directory('/home/dhaval/piyush/Usecases_dataset/voc_dataset_created/training_data',
                                                 target_size=(224,224),
                                                 color_mode='rgb',
                                                 batch_size=32,
                                                 class_mode='categorical',
                                                 shuffle=True)

validation_generator=train_datagen.flow_from_directory('/home/dhaval/piyush/Usecases_dataset/voc_dataset_created/validation_data',
                                                 target_size=(224,224),
                                                 color_mode='rgb',
                                                 batch_size=32,
                                                 class_mode='categorical',
                                                 shuffle=True)

step_size_train=train_generator.n//train_generator.batch_size
step_size_val=validation_generator.n//validation_generator.batch_size

tensorboard = TensorBoard(log_dir="logs/{}".format(time()), update_freq='epoch',profile_batch=0)

#fit the model
model.fit(
        train_generator,
        steps_per_epoch=step_size_train,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=step_size_val,
    callbacks= [tensorboard])

#model.fit(train_generator,steps_per_epoch=step_size_train,epochs=12)

model.save('mobilenet_model_voc_20class_ep_50_sgd.h5')








