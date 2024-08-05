import numpy as np
from Layers.FullyConnected import FullyConnected
from Layers.TanH import TanH
from Layers.Sigmoid import Sigmoid
from Layers.Base import BaseLayer

class RNN(BaseLayer):
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.trainable = True
        self.hidden_state = np.zeros((1, hidden_size))
        self.hidden_gradients = np.zeros((1, self.hidden_size))
        self._memorize = False
        self.tanh_activations = []
        self.sigmoid_activations = []
        
        # Initialize layers
        self.fc_hidden = FullyConnected(self.input_size + self.hidden_size, self.hidden_size)
        self.tanh = TanH()
        self.fc_output = FullyConnected(self.hidden_size, self.output_size)
        self.sigmoid = Sigmoid()
        
        self.optimizer = None
        self._gradient_weights = np.zeros_like(self.fc_hidden.gradient_weights)

    def forward(self, input_tensor):
       # print(f"Forward pass input_tensor shape: {input_tensor.shape}")
        self.input_tensor = input_tensor
        batch_size, sequence_length = input_tensor.shape
        self.output_tensor = np.zeros((batch_size, self.output_size))
        outputs = []
        self.combined_input_list = []
        # If memorize is False, reset hidden state
        if not self.memorize:
            self.hidden_state = np.zeros((1, self.hidden_size))
        
        for t in range(batch_size):
            #input_t = input_tensor[t,:].reshape(1, -1)  # Ensure input_t is 2D
            self.combined_input = np.concatenate(( self.hidden_state, self.input_tensor[t][None]), axis=1)
            #print(f"Combined input shape: {self.combined_input.shape}")  
            # self.combined_input_list.append(self.combined_input)
            # Compute the new hidden state
            hidden_input = self.fc_hidden.forward(self.combined_input)  # This includes W_h and b_h
            self.combined_input_list.append(self.fc_hidden.input_tensor)
            self.hidden_state = self.tanh.forward(hidden_input)
            self.tanh_activations.append(self.hidden_state)
            # Compute the output
            self.output_t = self.fc_output.forward(self.hidden_state)  # This includes W_hy and b_y
            self.output_tensor[t] = self.sigmoid.forward(self.output_t)
            self.sigmoid_activations.append(self.output_tensor[t])

        return self.output_tensor

    def backward(self, error_tensor):
        if self.memorize == False:
            self.hidden_gradients = np.zeros((1, self.hidden_size))

        self.error_tensor = error_tensor
        self.time_1 = self.error_tensor.shape[0]
        self.gradient_weights = np.zeros_like(self.fc_hidden.weights)
        self.output_error = np.zeros((self.time_1, self.input_size))

        for i, error in enumerate(reversed(error_tensor)):
            self.sigmoid.activations = self.sigmoid_activations[-i-1]
            self.fc_output.input_tensor = self.tanh_activations[-i-1]
            self.tanh.activations = self.tanh_activations[-i-1]
            self.fc_hidden.input_tensor = self.combined_input_list[-i-1]

            self.sigmoid_gradient = self.sigmoid.backward(error)
            self.fc2_gradient = self.fc_output.backward(self.sigmoid_gradient[None]) + self.hidden_gradients
            self.tanh_gradient = self.tanh.backward(self.fc2_gradient)
            self.fc1_gradient = self.fc_hidden.backward(self.tanh_gradient)


            self.output_error[-i - 1] = self.fc1_gradient[:, self.hidden_size:]
            self.hidden_gradients = self.fc1_gradient[:, :self.hidden_size]
            self.gradient_weights = self.gradient_weights + self.fc_hidden.gradient_weights

        if self.optimizer:
            self.weights = self.optimizer.calculate_update(self.weights, self.gradient_weights)

        return self.output_error


    

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, val):
        self._gradient_weights = val

    @property
    def weights(self):
        return self.fc_hidden.weights

    @weights.setter
    def weights(self, val):
        self.fc_hidden.weights = val

    @property
    def memorize(self):
        return self._memorize

    @memorize.setter
    def memorize(self, val):
        self._memorize = val

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, val):
        self._optimizer = val

    def initialize(self, weights_initializer, bias_initializer):
        self.fc_hidden.initialize(weights_initializer, bias_initializer)
        self.fc_output.initialize(weights_initializer, bias_initializer)
    
    def calculate_regularization_loss(self):
        return self.fc_hidden.calculate_regularization_loss() + self.fc_output.calculate_regularization_loss()