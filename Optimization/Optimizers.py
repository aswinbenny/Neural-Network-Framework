# Optimizers.py

import numpy as np
from .Constraints import L2_Regularizer, L1_Regularizer

class Optimizer:
    def __init__(self):
        self.regularizer = None

    def add_regularizer(self, regularizer):
        self.regularizer = regularizer

class Sgd(Optimizer):
    def __init__(self, learning_rate):
        super().__init__()
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        if self.regularizer:
            if isinstance(self.regularizer, L2_Regularizer):
                weight_tensor *= (1 - self.learning_rate * self.regularizer.alpha)
            elif isinstance(self.regularizer, L1_Regularizer):
                weight_tensor -= self.learning_rate * self.regularizer.alpha * np.sign(weight_tensor)
        return weight_tensor - self.learning_rate * gradient_tensor

class SgdWithMomentum(Optimizer):
    def __init__(self, learning_rate, momentum_rate):
        super().__init__()
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.velocity = None

    def calculate_update(self, weight_tensor, gradient_tensor):
        if self.velocity is None:
            self.velocity = np.zeros_like(weight_tensor)
        if self.regularizer:
            if isinstance(self.regularizer, L2_Regularizer):
                weight_tensor *= (1 - self.learning_rate * self.regularizer.alpha)
            elif isinstance(self.regularizer, L1_Regularizer):
                weight_tensor -= self.learning_rate * self.regularizer.alpha * np.sign(weight_tensor)
        self.velocity = self.momentum_rate * self.velocity - self.learning_rate * gradient_tensor
        return weight_tensor + self.velocity

class Adam(Optimizer):
    def __init__(self, learning_rate, mu, rho):
        super().__init__()
        self.learning_rate = learning_rate
        self.beta1 = mu
        self.beta2 = rho
        self.epsilon = np.finfo(float).eps
        self.m = None
        self.v = None
        self.t = 0

    def calculate_update(self, weight_tensor, gradient_tensor):
        if self.m is None:
            self.m = np.zeros_like(weight_tensor)
        if self.v is None:
            self.v = np.zeros_like(weight_tensor)
        
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradient_tensor
        self.v = self.beta2 * self.v + (1 - self.beta2) * (gradient_tensor ** 2)
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        
        if self.regularizer:
            if isinstance(self.regularizer, L2_Regularizer):
                weight_tensor *= (1 - self.learning_rate * self.regularizer.alpha)
            elif isinstance(self.regularizer, L1_Regularizer):
                weight_tensor -= self.learning_rate * self.regularizer.alpha * np.sign(weight_tensor)

        return weight_tensor - self.learning_rate * (m_hat / (np.sqrt(v_hat) + self.epsilon))