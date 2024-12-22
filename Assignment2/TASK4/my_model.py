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
        raise NotImplementedError("TO DO in my_model.py")
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
        # Apply layers here
        raise NotImplementedError("TO DO in my_model.py")
        # student code end

        return x