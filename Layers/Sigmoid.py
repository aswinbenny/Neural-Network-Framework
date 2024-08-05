import numpy as np
from Layers.Base import BaseLayer

class Sigmoid(BaseLayer):
    def __init__(self):
        self.activations = None
        self.trainable = False
    def forward(self, input_tensor):
        # Compute the Sigmoid activation
        self.activations = 1 / (1 + np.exp(-input_tensor))
        return self.activations

    def backward(self, error_tensor):
        # Compute the gradient of the loss w.r.t. the input tensor
        return error_tensor * self.activations * (1 - self.activations)