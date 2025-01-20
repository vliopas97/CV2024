# TUWIEN - WS2024 CV: Task4 - Mask Classification using CNN
# *********+++++++++*******++++INSERT GROUP NO. HERE

from typing import List
import torch
from torch.functional import Tensor
import torch.nn as nn
import torch.nn.functional as F


class MaskClassifier(nn.Module):

    def __init__(self, name, img_size=64, dropout: float = 0, batch_norm: bool = False):
        """
        Initializes the network architecture by creating a simple CNN of convolutional and max pooling layers.
        
        Args:
        - name: The name of the classifier.
        - img_size: Size of the input images.
        - dropout (float): Dropout rate between 0 and 1.
        - batch_norm (bool): Determines if batch normalization should be applied.
        """
        super(MaskClassifier, self).__init__()
        self.name = name
        self.img_size = img_size
        self.batch_norm = batch_norm

        # student code start
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * (img_size // 4) * (img_size // 4), 128)
        self.fc2 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(dropout)

        # student code end


    def forward(self, x: Tensor) -> Tensor:
        """
        Applies the predefined layers of the network to x.
        
        Args:
        - x (Tensor): Input tensor to be classified [batch_size x channels x height x width].
        
        Returns:
        - Tensor: Output tensor after passing through the network layers.
        """
        
        # student code start
        x = self.pool(F.relu(self.conv1(x)))  
        x = self.pool(F.relu(self.conv2(x)))  
        x = x.view(x.size(0), -1)  
        x = F.relu(self.fc1(x))   
        x = self.dropout(x)     
        x = torch.sigmoid(self.fc2(x))  
        # student code end

        return x