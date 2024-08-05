# FullyConnected.py

import numpy as np
from Layers.Base import BaseLayer

class FullyConnected(BaseLayer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.trainable = True
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.uniform(0, 1, (input_size + 1, output_size)) # add extra row for bias
        self._optimizer = None
        self._gradient_weights = None
        self.input_tensor = None

    def forward(self, input_tensor):
        # Add a column of ones to the input tensor for the bias
        batch_size = input_tensor.shape[0]
        augmented_input = np.hstack((input_tensor, np.ones((batch_size, 1))))
        self.input_tensor = augmented_input
        # Compute the output by performing the matrix multiplication
        self.output_tensor = np.dot(augmented_input, self.weights)
        return np.dot(self.input_tensor, self.weights)

    def backward(self, error_tensor):
        # Compute the gradient with respect to the weights
        self._gradient_weights = np.dot(self.input_tensor.T, error_tensor)
        # Compute the gradient with respect to the input tensor (excluding the bias column)
        output_error = np.dot(error_tensor, self.weights[:-1,:].T)

        # Update the weights if the optimizer is set
        if self._optimizer is not None:
            # if self._optimizer.regularizer:
            #     self._gradient_weights += self._optimizer.regularizer.calculate_gradient(self.weights)
            self.weights = self._optimizer.calculate_update(self.weights, self.gradient_weights)

        return output_error

    def initialize(self, weights_initializer, bias_initializer):
        self.weights[:-1, :] = weights_initializer.initialize((self.input_size, self.output_size), self.input_size, self.output_size)
        self.weights[-1, :] = bias_initializer.initialize((1, self.output_size), self.input_size, self.output_size)

    def calculate_regularization_loss(self):
        if self.optimizer and self.optimizer.regularizer:
            return self.optimizer.regularizer.norm(self.weights)
        return 0

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer

    @property
    def gradient_weights(self):
        return self._gradient_weights
    
    @gradient_weights.setter
    def gradient_weights(self, val):
        self._gradient_weights = val
