# neural_network.py

import copy
import pickle
import numpy as np

class NeuralNetwork:
    def __init__(self, optimizer, weights_initializer, bias_initializer):
        self.optimizer = optimizer
        self.loss = []
        self.layers = []
        self.data_layer = None
        self.loss_layer = None
        self.current_label = None
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer

    def forward(self):
        input_data, self.current_label = self.data_layer.next()
        output = input_data
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def backward(self):
        error_tensor = self.loss_layer.backward(self.current_label)
        for layer in reversed(self.layers):
            error_tensor = layer.backward(error_tensor)

    def append_layer(self, layer):
        if layer.trainable:
            if hasattr(layer, 'initialize'):
                layer.initialize(self.weights_initializer, self.bias_initializer)
            layer.optimizer = copy.deepcopy(self.optimizer)
        self.layers.append(layer)

    def compute_regularization_loss(self):
        reg_loss = 0.0
        for layer in self.layers:
            if layer.trainable:
                if layer.optimizer is not None:
                    if isinstance(layer.optimizer, tuple):
                        for optimizer_instance in layer.optimizer:
                            if optimizer_instance and hasattr(optimizer_instance,
                                                              'regularizer') and optimizer_instance.regularizer:
                                reg_loss += optimizer_instance.regularizer.norm(layer.weights)
                    #If it's not a tuple, check the regularizer directly
                    elif layer.optimizer and hasattr(layer.optimizer, 'regularizer') and layer.optimizer.regularizer:
                        reg_loss += layer.optimizer.regularizer.norm(layer.weights)
        return reg_loss

    def train(self, iterations):
        self.phase = False  # Set phase to training
        for _ in range(iterations):
            prediction = self.forward()
            data_loss = self.loss_layer.forward(prediction, self.current_label)
            # Calculate regularization loss
            reg_loss = self.compute_regularization_loss()
            total_loss = data_loss + reg_loss
            self.loss.append(total_loss)
            self.backward()

    def test(self, input_tensor):
        self.phase = True  # Set phase to testing
        output = input_tensor
        for layer in self.layers:
            output = layer.forward(output)
        return output

    @property
    def phase(self):
        if self.layers:
            return self.layers[0].phase
        return False

    @phase.setter
    def phase(self, phase):
        for layer in self.layers:
            if hasattr(layer, 'phase'):
                layer.phase = phase
    
    def __getstate__(self):
        state = self.__dict__.copy()
        state['data_layer'] = None  # Exclude data layer from being saved
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.data_layer = None  # Initialize dropped members with None

def save(filename, net):
    with open(filename, 'wb') as f:
        pickle.dump(net, f)

def load(filename, data_layer):
    with open(filename, 'rb') as f:
        net = pickle.load(f)
        net.data_layer = data_layer  # Set the data layer again
    return net