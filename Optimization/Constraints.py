# Constraints.py

import numpy as np

class L2_Regularizer:
    def __init__(self, alpha):
        self.alpha = alpha

    def calculate_gradient(self, weights):
        self.weights = weights
        return self.alpha * self.weights  

    def norm(self, weights):
        self.weights = weights
        return self.alpha * np.sum(self.weights ** 2)

class L1_Regularizer:
    def __init__(self, alpha):
        self.alpha = alpha

    def calculate_gradient(self, weights):
        self.weights = weights
        return self.alpha * np.sign(self.weights)

    def norm(self, weights):
        self.weights = weights
        return self.alpha * np.sum(np.abs(self.weights))