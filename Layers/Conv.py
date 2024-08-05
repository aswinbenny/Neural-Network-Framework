# Conv.py

import numpy as np
from scipy.signal import correlate, convolve
import copy
from .Base import BaseLayer

class Conv(BaseLayer):
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        super().__init__()
        self.trainable = True
        self.stride_shape = stride_shape
        self.convolution_shape = convolution_shape
        self.num_kernels = num_kernels
        self.weights = np.random.uniform(0, 1, (self.num_kernels, *self.convolution_shape))
        self.bias = np.random.uniform(0, 1, self.num_kernels)
        self._gradient_weights = None
        self._gradient_bias = None
        self._weights_optimizer = None
        self._bias_optimizer = None

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @property
    def gradient_bias(self):
        return self._gradient_bias

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        batch = input_tensor.shape[0]
        conv_output = np.zeros((batch, self.num_kernels) + input_tensor.shape[2:])

        if len(input_tensor.shape) == 3:  # 1D convolution
            for i in range(batch):
                for j in range(self.num_kernels):
                    output = correlate(input_tensor[i], self.weights[j], mode='same', method='direct')
                    center_value = output[self.convolution_shape[0] // 2]
                    conv_output[i, j] = center_value + self.bias[j]
            conv_output = conv_output[:, :, ::self.stride_shape[0]]
        else:  # 2D convolution
            for i in range(batch):
                for j in range(self.num_kernels):
                    output = correlate(input_tensor[i], self.weights[j], mode='same', method='direct')
                    center_value = output[self.convolution_shape[0] // 2]
                    conv_output[i, j] = center_value + self.bias[j]
            conv_output = conv_output[:, :, ::self.stride_shape[0], ::self.stride_shape[1]]

        return conv_output

    @property
    def optimizer(self):
        return self._weights_optimizer, self._bias_optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._weights_optimizer = optimizer
        self._bias_optimizer = copy.deepcopy(optimizer)

    def backward(self, error_tensor):
        batch = self.input_tensor.shape[0]
        gradient_tensor = np.zeros_like(self.input_tensor)
        errors = np.zeros((batch, self.num_kernels) + self.input_tensor.shape[2:])
        transposed_weights = np.transpose(self.weights, (1, 0, *range(2, self.weights.ndim)))
        flipped_weights = np.flip(transposed_weights, 1)

        if len(self.input_tensor.shape) == 3:  # 1D convolution
            errors[:, :, ::self.stride_shape[0]] = error_tensor
            pad = self.convolution_shape[1] // 2
            for i in range(batch):
                for j in range(transposed_weights.shape[0]):
                    gradient_tensor[i, j] = convolve(errors[i], flipped_weights[j], mode='same')[self.convolution_shape[0] // 2]
            self._gradient_weights = np.zeros_like(self.weights)

            for i in range(batch):
                for j in range(self.num_kernels):
                    pad_values = (pad, pad - 1 + (self.convolution_shape[1] % 2))
                    input_padded = np.pad(self.input_tensor[i], pad_values, 'constant', constant_values=0)
                    for m in range(self.input_tensor.shape[1]):
                        self._gradient_weights[j, m] += correlate(input_padded[m], errors[i, j], mode='valid')
            self._gradient_bias = np.sum(error_tensor, axis=(0, 2))

        else:  # 2D convolution
            errors[:, :, ::self.stride_shape[0], ::self.stride_shape[1]] = error_tensor
            pad1 = self.convolution_shape[1] // 2
            pad2 = self.convolution_shape[2] // 2

            for i in range(batch):
                for j in range(transposed_weights.shape[0]):
                    pad_values = [(0, 0), (pad1, pad1 - 1 + (self.convolution_shape[1] % 2)), (pad2, pad2 - 1 + (self.convolution_shape[2] % 2))]
                    error_padded = np.pad(errors[i], pad_values, mode='constant', constant_values=0)
                    gradient_tensor[i, j] = convolve(error_padded, flipped_weights[j], mode='valid')
            self._gradient_weights = np.zeros_like(self.weights)

            for i in range(batch):
                for j in range(self.num_kernels):
                    pad_values = [(0, 0), (pad1, pad1 - 1 + (self.convolution_shape[1] % 2)), (pad2, pad2 - 1 + (self.convolution_shape[2] % 2))]
                    input_padded = np.pad(self.input_tensor[i], pad_values, mode='constant', constant_values=0)
                    for m in range(self.input_tensor.shape[1]):
                        self._gradient_weights[j, m] += correlate(input_padded[m], errors[i, j], mode='valid')
            self._gradient_bias = np.sum(error_tensor, axis=(0, 2, 3))

        if self._weights_optimizer is not None:
            if self._weights_optimizer.regularizer:
                self._gradient_weights += self._weights_optimizer.regularizer.calculate_gradient(self.weights)
            self.weights = self._weights_optimizer.calculate_update(self.weights, self._gradient_weights)

        if self._bias_optimizer is not None:
            self.bias = self._bias_optimizer.calculate_update(self.bias, self._gradient_bias)

        return gradient_tensor

    def initialize(self, weights_initializer, bias_initializer):
        fan_in = np.prod(self.convolution_shape[:3])
        fan_out = self.num_kernels * np.prod(self.convolution_shape[1:])
        self.weights = weights_initializer.initialize(self.weights.shape, fan_in, fan_out=fan_out)
        self.bias = bias_initializer.initialize(self.bias.shape, fan_in=fan_in, fan_out=fan_out)

    def calculate_regularization_loss(self):
        if self._weights_optimizer and self._weights_optimizer.regularizer:
            return self._weights_optimizer.regularizer.norm(self.weights)
        return 0