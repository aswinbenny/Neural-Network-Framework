# Models/LeNet.py

from Layers.Conv import Conv
from Layers.FullyConnected import FullyConnected
from Layers.ReLU import ReLU
from Layers.SoftMax import SoftMax
from Layers.Flatten import Flatten
from Layers.Pooling import Pooling
from Optimization.Optimizers import Adam
from Optimization.Constraints import L2_Regularizer
from NeuralNetwork import NeuralNetwork

def build():
    # Initialize the network with Adam optimizer and L2 regularizer
    optimizer = Adam(learning_rate=5e-4)
    optimizer.add_regularizer(L2_Regularizer(4e-4))

    net = NeuralNetwork(optimizer)
    
    # Convolutional and subsampling layers
    net.append_layer(Conv((1, 6), (5, 5), 6, padding='same'))
    net.append_layer(ReLU())
    net.append_layer(Pooling((2, 2), (2, 2)))

    net.append_layer(Conv((6, 16), (5, 5), 16, padding='same'))
    net.append_layer(ReLU())
    net.append_layer(Pooling((2, 2), (2, 2)))

    # Flatten layer to convert 3D feature maps to 1D feature vectors
    net.append_layer(Flatten())

    # Fully connected layers
    net.append_layer(FullyConnected(400, 120))
    net.append_layer(ReLU())

    net.append_layer(FullyConnected(120, 84))
    net.append_layer(ReLU())

    net.append_layer(FullyConnected(84, 10))

    # SoftMax layer
    net.append_layer(SoftMax())

    return net