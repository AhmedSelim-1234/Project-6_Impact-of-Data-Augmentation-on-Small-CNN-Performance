import numpy as np
import torch
import torch.nn as nn

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(CNNBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.block(x)

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        
        # Layer 1
        self.conv_block1 = CNNBlock(in_channels=3, out_channels=32)

        # Layer 2
        self.conv_block2 = CNNBlock(in_channels=32, out_channels=64)

        # Layer 3
        self.conv_block3 = CNNBlock(in_channels=64, out_channels=128)

        # Layer 4: fully connected classifier block.
        self.classifier = nn.Sequential(
            nn.Flatten(),

            nn.Dropout(0.5),
            # Direct mapping from flattened features to output classes (4th weighted layer)
            nn.Linear(128 * 4 * 4, num_classes),
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.classifier(x)
        return x
