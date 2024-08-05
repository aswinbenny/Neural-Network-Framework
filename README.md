# Deep Learning Regularization and Recurrent Layers

## Overview

This project provides a modular and extensible framework for implementing and experimenting with various deep learning techniques, particularly focusing on regularization strategies and recurrent layers. The framework is designed to be flexible, enabling easy integration of new components and facilitating the construction of different neural network architectures.

## Features

- **Regularization Techniques**:
  - Implementation of L1 and L2 regularization strategies to prevent overfitting.
  - Incorporation of Dropout as a regularization method, specifically for fully connected layers.

- **Layer Implementations**:
  - Modular implementations of core layers, including Convolutional, Fully Connected, Batch Normalization, and various activation functions like ReLU, Sigmoid, and TanH.
  - Recurrent Neural Networks (RNNs) with support for Elman networks and Long Short-Term Memory (LSTM) cells.

- **Optimization**:
  - Custom optimizers with support for integrating regularization techniques.
  - Batch Normalization and its variant for convolutional layers.

- **Neural Network Architecture**:
  - Implementation of a variant of the LeNet architecture.
  - Support for saving and loading models using Python's pickle module.

- **Testing and Debugging**:
  - Comprehensive test suite to validate the correctness of the implementations.
  - Support for debugging and bonus computation through automated test scripts.

## Project Structure

```plaintext
.
├── __pycache__/
├── Layers/
│   ├── __init__.py
│   ├── Base.py
│   ├── BatchNormalization.py
│   ├── Conv.py
│   ├── Dropout.py
│   ├── Flatten.py
│   ├── FullyConnected.py
│   ├── Helpers.py
│   ├── Initializers.py
│   ├── Pooling.py
│   ├── ReLU.py
│   ├── RNN.py
│   ├── Sigmoid.py
│   ├── SoftMax.py
│   ├── TanH.py
├── Models/
│   ├── __init__.py
│   ├── LeNet.py
├── Optimization/
│   ├── __pycache__/
│   ├── __init__.py
│   ├── Constraints.py
│   ├── Loss.py
│   ├── Optimizers.py
├── log.txt
├── NeuralNetwork.py
├── NeuralNetworkTests.py
├── TrainLeNet.py