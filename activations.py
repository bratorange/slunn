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


class Softmax(Module):
    def optimize(self, learning_rate):
        pass

    def forward(self, x):
        exps = np.exp(x - np.max(x))
        self.y = exps / np.sum(exps)
        return self.y

    def backward(self, gradient_y):
        jacobian = np.diag(self.y) - np.outer(self.y, self.y)

        # Compute gradient
        gradient_x = np.dot(jacobian, gradient_y)
        self.y = None
        return gradient_x
