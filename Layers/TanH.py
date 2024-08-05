import numpy as np
from Layers.Base import BaseLayer

class TanH(BaseLayer):
    def __init__(self):
        self.activations = None
        self.trainable = False

    def forward(self, input_tensor):
        # Compute the TanH activation
        self.activations = np.tanh(input_tensor)
        return self.activations

    def backward(self, error_tensor):
        # Compute the gradient of the loss w.r.t. the input tensor
        return error_tensor * (1 - self.activations ** 2)