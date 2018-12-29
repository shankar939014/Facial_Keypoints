## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        # convolutional layer (sees 224x224x1 tensor)
        # size of the imput image=W-F+2P/S+1
        self.conv1 = nn.Conv2d(1, 16, 5)
        self.bn1= nn.BatchNorm2d(16)
        # convolutional layer (sees 111x111x16 tensor)
        self.conv2 = nn.Conv2d(16, 32, 5, padding=1)
        self.bn2= nn.BatchNorm2d(32)
        # convolutional layer (sees 54x54x32 tensor)
        self.conv3 = nn.Conv2d(32, 64, 5, padding=1)
        self.bn3= nn.BatchNorm2d(64)
        # convolutional layer (sees 26x26x64 tensor)
        self.conv4 = nn.Conv2d(64, 128, 5, padding=1)
        self.bn4= nn.BatchNorm2d(128)
        # convolutional layer (sees 12x12x128 tensor)
        self.conv5 = nn.Conv2d(128, 256, 5, padding=1)
        self.bn5= nn.BatchNorm2d(256)
        # max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        # linear layer (256 * 5 * 5 -> 1024)
        self.fc1 = nn.Linear(256 * 5 * 5 , 1024)
        # linear layer (1024 -> 136)
        self.fc2 = nn.Linear(1024, 136)
        # dropout layer (p=0.25)
        self.dropout = nn.Dropout(0.25)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.pool(F.relu(self.bn5(self.conv5(x))))
       
        
        # flatten image input
        x = x.view(-1, 256 * 5 * 5)
        # add dropout layer
        x = self.dropout(x)
        # add 1st hidden layer, with relu activation function
        x = F.relu(self.fc1(x))
        # add dropout layer
        x = self.dropout(x)
        # add 2nd hidden layer, with relu activation function
        x = self.fc2(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
