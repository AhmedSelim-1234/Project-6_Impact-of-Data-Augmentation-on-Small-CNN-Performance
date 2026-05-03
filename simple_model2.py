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
        # 1st convolutional block
        self.conv_block1 = CNNBlock(in_channels=3, out_channels=32)

        # 2nd convolutional block
        self.conv_block2 = CNNBlock(in_channels=32, out_channels=64)

        # 3rd convolutional block
        self.conv_block3 = CNNBlock(in_channels=64, out_channels=128)

        # 4th convolutional block 
        self.conv_block4 = CNNBlock(in_channels=128, out_channels=256)

        # --- REPLACING FULLY CONNECTED LAYERS ---
        
        # 1x1 Convolution to map the 256 channels directly to the number of classes
        self.final_conv = nn.Conv2d(in_channels=256, out_channels=num_classes, kernel_size=1)
        
        # Global Average Pooling to average out the spatial dimensions (Height & Width)
        # AdaptiveAvgPool2d((1, 1)) guarantees the spatial output will be exactly 1x1
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        # Pass input through the feature extractors
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x) 

        # Map to class scores using the 1x1 convolution
        # Output shape: (Batch_Size, num_classes, Height, Width)
        x = self.final_conv(x)
        
        # Pool the spatial dimensions down to a single pixel
        # Output shape: (Batch_Size, num_classes, 1, 1)
        x = self.global_pool(x)
        
        # Flatten the remaining 1x1 spatial dimensions to get a standard 1D output
        # Output shape: (Batch_Size, num_classes)
        x = torch.flatten(x, 1) 

        return x