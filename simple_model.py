import numpy as np
import torch
import torch.nn as nn

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(CNNBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            #This layer needs to know the number of channels of its input, which is the output of the previous convolutional layer.
            nn.BatchNorm2d(out_channels), # # Makes training more stable and faster
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.block(x)

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        #first convolutional block
        self.conv_block1 = CNNBlock(in_channels=3, out_channels=32)

        #second convolutional block
        self.conv_block2 = CNNBlock(in_channels=32, out_channels=64)

        #third convolutional block
        self.conv_block3 = CNNBlock(in_channels=64, out_channels=128)

        # Define the fully connected classifier block.
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(),
            # dropout layer to prevent overfitting by randomly setting a fraction of inputs to zero.
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        #pass input through the first convolutional block
        x = self.conv_block1(x)

        #pass feature maps through the second convolutional block
        x = self.conv_block2(x)

        #pass feature maps through the third convolutional block
        x = self.conv_block3(x)

        # Flatten the output for the fully connected layers
        # Pass the flattened features through the fully connected layers
        x = self.classifier(x)

        return x