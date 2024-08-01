import numpy as np

from Module import Module


class ReLU(Module):
    def optimize(self, learning_rate):
        self.x = None

    def forward(self, x):
        self.x = x
        return np.maximum(0, x)

    def backward(self, gradient_y):
        assert self.x is not None
        return np.where(self.x < 0., 0., 1.) * gradient_y
