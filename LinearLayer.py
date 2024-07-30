import numpy as np

import settings
from Module import Module


# I want to limit the input to vectors. Maybe tensor generalisation comes later
class LinearLayer(Module):
    def __init__(self, input_size, output_size, weights=np.ndarray):
        self.weights = np.zeros((input_size, output_size), settings.default_dtype)
        self.bias = np.zeros(output_size, settings.default_dtype)

    def forward(self, x):
        assert type(x) == np.ndarray
        assert x.shape[-1] == self.weights.shape[0]
        assert x.dtype == settings.default_dtype

        # save the input for the backward pass
        self.x = x
        return np.dot(self.x, self.weights) + self.bias

    def get_parameters(self):
        return {"weights": self.weights, "bias": self.bias}

    def set_parameters(self, dict):
        self.weights = dict["weights"]
        self.bias = dict["bias"]

    def randomize(self):
        fan_in = self.weights.shape[0]
        std = np.sqrt(2.0 / fan_in)

        self.weights = np.random.normal(0, std, self.weights.shape)
        self.bias = np.full(self.bias.shape, fill_value=0., dtype=settings.default_dtype)

    def backward(self, gradient_y):
        # gradient_output is the gradient of the loss with respect to the output of this layer
        # gradient_input is the gradient of the loss with respect to the input of this layer
        assert type(gradient_y) == np.ndarray
        assert self.x is not None
        assert gradient_y.shape[-1] == self.weights.shape[1]
        assert gradient_y.dtype == settings.default_dtype

        # store the gradients regarding weights and biases in the LinearLayer class
        self.gradient_weights = np.einsum('bi,bj->bij', self.x, gradient_y)
        self.gradient_weights = self.gradient_weights.mean(axis=0)
        self.gradient_bias = gradient_y.mean(axis=0)
        self.gradient_x = np.dot(gradient_y, self.weights.T)

        return self.gradient_x

    def optimize(self, learning_rate):
        assert self.gradient_bias is not None
        assert self.gradient_bias is not None

        self.weights -= learning_rate * self.gradient_weights
        self.bias -= learning_rate * self.gradient_bias

        # reset the gradients
        self.gradient_weights = None
        self.gradient_bias = None
