import numpy as np
from Layers.Base import BaseLayer

class BatchNormalization(BaseLayer):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.epsilon = np.finfo(float).eps,
        self.momentum = 0.8
        self.trainable = True
        self.weights = np.ones(channels)
        self.bias = np.zeros(channels)
        self.mean = np.zeros(channels)
        self.variance = np.ones(channels)
        self.moving_mean = np.zeros(channels)
        self.moving_variance = np.ones(channels)
        self._gradient_weights = None
        self._gradient_bias = None
        self.optimizer = None

    def initialize(self, weights_initializer = None, bias_initializer = None):
        self.weights = np.ones(self.channels)
        self.bias = np.zeros(self.channels)

    def forward(self, input_tensor):
        self.input_tensor = input_tensor  # Store input tensor

        if len(input_tensor.shape) == 4:  # image-like
            input_tensor = self.reformat(input_tensor)

        if not self.testing_phase:
            self.mean = np.mean(input_tensor, axis=0)
            self.variance = np.var(input_tensor, axis=0)
            self.moving_mean = self.momentum * self.mean + (1 - self.momentum) * self.mean
            self.moving_variance = self.momentum * self.variance + (1 - self.momentum) * self.variance
        else:
            self.mean = self.moving_mean
            self.variance = self.moving_variance

        self.normalized_input = (input_tensor - self.mean) / np.sqrt(self.variance + self.epsilon)
        output = self.weights.reshape(1, -1) * self.normalized_input + self.bias.reshape(1, -1)

        if len(output.shape) == 2 and len(self.input_tensor.shape) == 4:  # reformat back to image-like
            output = self.reformat(output)

        return output

    def backward(self, error_tensor):
        if len(error_tensor.shape) == 4:  # image-like
            error_tensor = self.reformat(error_tensor)

        batch_size = error_tensor.shape[0]

        self._gradient_weights = np.sum(error_tensor * self.normalized_input, axis=0)
        self._gradient_bias = np.sum(error_tensor, axis=0)
        grad_normalized = error_tensor * self.weights.reshape(1, -1)

        grad_input = (1. / batch_size) * (1. / np.sqrt(self.variance + self.epsilon)) * \
                     (batch_size * grad_normalized - np.sum(grad_normalized, axis=0) - 
                      self.normalized_input * np.sum(grad_normalized * self.normalized_input, axis=0))

        if len(grad_input.shape) == 2 and len(self.input_tensor.shape) == 4:  # reformat back to image-like
            grad_input = self.reformat(grad_input)

        if self.optimizer:
            self.weights = self.optimizer.calculate_update(self.weights, self._gradient_weights)
            self.bias = self.optimizer.calculate_update(self.bias, self._gradient_bias)

        return grad_input

    def reformat(self, tensor):
        if len(tensor.shape) == 4:  # image-like
            B, C, H, W = tensor.shape
            return tensor.transpose(0, 2, 3, 1).reshape(B * H * W, C)
        elif len(tensor.shape) == 2:  # vector-like
            B, C, H, W = self.input_tensor.shape
            return tensor.reshape(B, H, W, C).transpose(0, 3, 1, 2)
        else:
            raise ValueError("Unsupported tensor shape for reformat")

    def calculate_regularization_loss(self):
        return 0  # no regularization for BatchNormalization

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @property
    def gradient_bias(self):
        return self._gradient_bias

    @gradient_weights.setter
    def gradient_weights(self, gradients):
        self._gradient_weights = gradients

    @gradient_bias.setter
    def gradient_bias(self, gradients):
        self._gradient_bias = gradients