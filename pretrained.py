"""
pretrained.py adds onto and fine-tunes the InceptionV3 model for our 17 or 102 flower datasets

This trains a convolutional neural network using transfer learning to take advantage of InceptionV3's architecture
An advantage of fine-tuning a pretrained network is using its extensive feature detection capabilites
from it training on the 14.5 million image dataset of ImageNet

There will be two datasets of flowers that we can choose from:
17 flowers: has 80 images per class (17 classes) and we split it with 70 images train and 10 images for validation
102 flowers: has 50 to 250 images per class (102 classes) split into train with 75% images and validation 25% for each

102 flowers was able to achieve an accuracy of around 95% after training for about 6 hours
17 flowers was able to achieve an accuracy of around 98% after training for about 2 hours

With very little data (relatively) we were able to train two different models that were highly accurate
"""

from keras.applications.inception_v3 import InceptionV3
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D

# Create the base pre-trained model
from keras.preprocessing.image import ImageDataGenerator

base_model = InceptionV3(weights='imagenet', include_top=False)
batch_size = 32  # batch size is chosen to help keep GPU memory usage low
dim = 299  # InceptionV3 is trained on 299x299 images

num_training_img = 6182  # number of training images for 102 dataset
# num_training_img = 1190 # number of training images for 17 dataset
num_val_img = 2009  # number of validation images for 102 dataset
# num_val_img = 170 # number of validation images for 17 dataset
# Add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# Add a fully-connected layer
x = Dense(1024, activation='relu')(x)

# Select 102 or 17 for whichever model you want to train on.
predictions = Dense(102, activation='softmax')(x)
# predictions = Dense(17, activation='softmax')(x)

# This is the model we will train
model = Model(input=base_model.input, output=predictions)

# First: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# Compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# preprocessor for the training training data
train_datagen = ImageDataGenerator(
    rescale=1. / 255,  # regularize RGB color values to floats between 0.0 to 1.0
    horizontal_flip=True,
    fill_mode='constant')  # in case resizing needs to create new image - fill it in with black

# preprocessor for the testing data
# only rescaling for regularization
test_datagen = ImageDataGenerator(rescale=1. / 255)

# generator for creating batch_size of training images with all transformations applied
# leaves us with more images than we start with
train_generator = train_datagen.flow_from_directory(
    '102_flowers/train',  # this is the target directory
    target_size=(dim, dim),  # all images will be resized to dim * dim
    batch_size=batch_size,
    class_mode='categorical')  # since loss = categorical_crossentropy, our class labels should be categorical

# generator for creating batch_size of validation images with only regularization and resizing
# leaves us with the same amount of images that we start with
validation_generator = test_datagen.flow_from_directory(
    '102_flowers/validation',
    target_size=(dim, dim),
    batch_size=batch_size,
    class_mode='categorical')
# class labels are usually alphabetical and sequential, but just in case print out the class labels as as dictionary
print(validation_generator.class_indices)

# all InceptionV3 layers are frozen, we train our added layers, where the weights were randomly initialized
# only need to go through a couple of epochs

model.fit_generator(
    train_generator,
    steps_per_epoch=num_training_img // batch_size,  # steps =  num_images // batch_size = total num of complete passes
    epochs=2,
    validation_data=validation_generator,
    validation_steps=num_val_img // batch_size)

# At this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.

# Let's visualize layer names and layer indices to see how many layers we should freeze:
for i, layer in enumerate(base_model.layers):
    print(i, layer.name)

# We chose to train the top 2 inception blocks, i.e. we will freeze
# the first 172 layers and unfreeze the rest. This prevents overfitting.
for layer in model.layers[:172]:
    layer.trainable = False
for layer in model.layers[172:]:
    layer.trainable = True

# We need to recompile the model for these modifications to take effect
# We use SGD with a low learning rate
from keras.optimizers import SGD

model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

# train our model while back propagating through our layers and the top two blocks of the InceptionV3 architecture
model.fit_generator(
    train_generator,
    steps_per_epoch=num_training_img // batch_size,
    epochs=200,
    validation_data=validation_generator,
    validation_steps=num_val_img // batch_size,
    callbacks=[EarlyStopping(patience=50), ModelCheckpoint('102_model3.h5', verbose=1, save_best_only=True)])
