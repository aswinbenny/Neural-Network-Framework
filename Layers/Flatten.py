import numpy as np
from Layers.Base import BaseLayer

class Flatten(BaseLayer):
    def __init__(self):
        self.input_shape = None
        self.trainable = False

    def forward(self, input_tensor):
        # Save the original shape to use in the backward pass
        self.input_shape = input_tensor.shape
        # Flatten the input tensor, preserving the batch size
        return input_tensor.reshape(self.input_shape[0], -1)

    def backward(self, error_tensor):
        # Reshape the error tensor back to the original input shape
        return error_tensor.reshape(self.input_shape)
