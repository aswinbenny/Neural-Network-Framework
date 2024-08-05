# Dropout.py

import numpy as np
from Layers.Base import BaseLayer

class Dropout(BaseLayer):
    def __init__(self, probability):
        self.probability = probability
        self.mask = None
        self.testing_phase = False
        self.trainable = False

    def forward(self, input_tensor):
        if not self.testing_phase:
            self.mask = np.random.binomial(1, self.probability, size=input_tensor.shape) / self.probability
            return input_tensor * self.mask
        else:
            return input_tensor

    def backward(self, error_tensor):
        if self.mask is not None:
            return error_tensor * self.mask
        else:
            return error_tensor

    @property
    def phase(self):
        return self.testing_phase

    @phase.setter
    def phase(self, phase):
        self.testing_phase = phase