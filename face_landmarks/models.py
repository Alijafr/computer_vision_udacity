## TODO: define the convolutional neural network architecture

import torch
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
        # 1 input image channel (grayscale), 32 output channels/feature maps, 4x4 square convolution kernel
        
        #following paper model
        #assume input size is (224,224)
        self.conv1 = nn.Conv2d(1, 32, 5) 
        self.conv2 = nn.Conv2d(32,64, 5)
        self.conv3 = nn.Conv2d(64,128,5)
        self.conv4 = nn.Conv2d(128,256,5)
        #output after pooling = (10,10,256)
        self.fc1 = nn.Linear(10*10*256, 1500)
        self.fc2 = nn.Linear(1500, 1500)
        self.fc3 = nn.Linear(1500, 136)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        self.pool= nn.MaxPool2d(2,2)
        self.dropout1 = nn.Dropout(p=0.1)
#         self.dropout2 = nn.Dropout(p=0.2)
#         self.dropout3 = nn.Dropout(p=0.3)
#         self.dropout4 = nn.Dropout(p=0.4)
#         self.dropout5 = nn.Dropout(p=0.5)
#         self.dropout6 = nn.Dropout(p=0.6)
        
        #batch norms
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.batch_norm3 = nn.BatchNorm2d(128)
        self.batch_norm4 = nn.BatchNorm2d(256)

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        
        #option one: use only dropout
#         x = self.dropout1(self.pool(F.relu(self.conv1(x))))
#         x = self.dropout2(self.pool(F.relu(self.conv2(x))))
#         x = self.dropout3(self.pool(F.relu(self.conv3(x))))
#         x = self.dropout4(self.pool(F.relu(self.conv4(x))))
        
        #optoin 2: use only batch norms
        
#         x = self.pool(self.batch_norm1(F.relu(self.conv1(x))))
#         x = self.pool(self.batch_norm2(F.relu(self.conv2(x))))
#         x = self.pool(self.batch_norm3(F.relu(self.conv3(x))))
#         x = self.pool(self.batch_norm4(F.relu(self.conv4(x))))
        
        #option 3: combine batch notms and dropout with same rate p=0.1
        
        x = self.dropout1(self.pool(self.batch_norm1(F.relu(self.conv1(x)))))
        x = self.dropout1(self.pool(self.batch_norm2(F.relu(self.conv2(x)))))
        x = self.dropout1(self.pool(self.batch_norm3(F.relu(self.conv3(x)))))
        x = self.dropout1(self.pool(self.batch_norm4(F.relu(self.conv4(x)))))
        
        x = x.view(x.size(0), -1) #flatten 
        x = self.dropout1(F.relu(self.fc1(x)))
        x = self.dropout1(F.relu(self.fc2(x)))
        x = self.fc3(x)
        
        
        
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
