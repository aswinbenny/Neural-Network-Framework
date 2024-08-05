import numpy as np
from Layers.Base import BaseLayer

class ReLU(BaseLayer):
    def __init__(self):
        self.input_tensor = None
        super().__init__()

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        return np.maximum(0, input_tensor)

    def backward(self, error_tensor):
        # Derivative of ReLU function: 1 for positive values, 0 for negative values
        gradient_input = error_tensor * (self.input_tensor > 0).astype(float)
        return gradient_input

