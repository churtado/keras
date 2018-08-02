from keras.datasets import mnist
from keras.utils import to_categorical

import torch
import torch.nn as nn
import torch.nn.functional as F

num_epochs = 5
batch_size = 128

image_width = 28
image_height = 28

# get the data set
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# the data is 60k matrices. Reshape each image to a long array of (28*28) numbers
train_images = train_images.reshape((60000, image_width * image_height))
# gray-scale values go from 0-255. Change so they go from 0-1
train_images = train_images.astype('float32') / 255

# do same for test set
test_images = test_images.reshape((10000, image_width * image_height))
test_images = test_images.astype('float32') / 255


# each element in these tensors is a number from 0-9. Change so they are 10-dim, 1-axis binary tensor
# with only 1 on the value pertaining to the class. This is known as 1-hot encoding
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)


#####################################################


train_images = torch.tensor(train_images)
train_labels = torch.tensor(train_labels)

test_images = torch.tensor(test_images)
test_labels = torch.tensor(test_labels)



class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.fc1 = nn.Linear(512, 10)
        nn.Linear()
        self.fc2 = nn.Linear(10, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

