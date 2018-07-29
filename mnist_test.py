from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical

num_epochs = 5
batch_size = 128

image_width = 28
image_height = 28

# get the data set
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# build the network
network = models.Sequential()

# dense means densely connected. In other words, all nodes are connected to all on next layer
network.add(layers.Dense(512, activation='relu', input_shape=(image_width * image_height,)))
network.add(layers.Dense(10, activation='softmax'))

network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])


# the data is 60k matrices. Reshape each image to a long array of (28*28) numbers
train_images = train_images.reshape((60000, image_width * image_height))
# gray-scale values go from 0-255. Change so they go from 0-1
train_images = train_images.astype('float32') / 255

# do same for test set
test_images = test_images.reshape((10000, image_width * image_height))
test_images = test_images.astype('float32') / 255


# each element in these tensors is a number from 0-9. Change so they are 10-dim, 1-axis binary tensor
# with only 1 on the value pertaining to the class
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# train the network
network.fit(train_images, train_labels, epochs=num_epochs, batch_size=batch_size)

# test the network
test_loss, test_acc = network.evaluate(test_images, test_labels)

print('test_acc:', test_acc)