import numpy as np
from Layers.Base import BaseLayer


class SoftMax(BaseLayer):
    def __init__(self):
        super().__init__()
        self.input_tensor = None
        self.softmax_probs = None

    def forward(self, input_tensor):
        # Subtract the maximum value for numerical stability
        exp_values = np.exp(input_tensor - np.max(input_tensor, axis=1, keepdims=True))
        self.input_tensor = input_tensor
        # Compute softmax probabilities
        softmax_out = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.softmax_probs = softmax_out
        return softmax_out

    def backward(self, error_tensor):
        # Compute the sum of the element-wise product of error_tensor and softmax_output
        sum_term = np.sum(error_tensor * self.softmax_probs, axis=1, keepdims=True)
        # Compute the gradient of the loss with respect to the input of the SoftMax layer
        gradient = self.softmax_probs * (error_tensor - sum_term)
        return gradient
