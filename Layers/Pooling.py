import numpy as np
from Layers.Base import BaseLayer

class Pooling(BaseLayer):
    def __init__(self, stride_shape, pooling_shape):
        self.stride_shape = (stride_shape, stride_shape) if isinstance(stride_shape, int) else stride_shape
        self.pooling_shape = (pooling_shape, pooling_shape) if isinstance(pooling_shape, int) else pooling_shape
        self.input_tensor = None
        self.max_indices = None
        self.trainable = False

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        batch_size, channels, input_height, input_width = input_tensor.shape
        pool_height, pool_width = self.pooling_shape
        stride_height, stride_width = self.stride_shape
        
        # Calculate output dimensions
        output_height = (input_height - pool_height) // stride_height + 1
        output_width = (input_width - pool_width) // stride_width + 1
        
        # Initialize output tensor
        output_tensor = np.zeros((batch_size, channels, output_height, output_width))
        self.max_indices = np.zeros((batch_size, channels, output_height, output_width, 2), dtype=int)
        
        for b in range(batch_size):
            for c in range(channels):
                for i in range(output_height):
                    for j in range(output_width):
                        start_i = i * stride_height
                        start_j = j * stride_width
                        end_i = start_i + pool_height
                        end_j = start_j + pool_width
                        
                        pooling_region = input_tensor[b, c, start_i:end_i, start_j:end_j]
                        max_value = np.max(pooling_region)
                        output_tensor[b, c, i, j] = max_value
                        
                        max_index = np.unravel_index(np.argmax(pooling_region), pooling_region.shape)
                        self.max_indices[b, c, i, j] = [start_i + max_index[0], start_j + max_index[1]]
                        
        return output_tensor

    def backward(self, error_tensor):
        batch_size, channels, output_height, output_width = error_tensor.shape
        output_error = np.zeros_like(self.input_tensor)
        
        for b in range(batch_size):
            for c in range(channels):
                for i in range(output_height):
                    for j in range(output_width):
                        max_i, max_j = self.max_indices[b, c, i, j]
                        output_error[b, c, max_i, max_j] += error_tensor[b, c, i, j]
                        
        return output_error