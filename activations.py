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
        # Ensure gradient_y is 2D: (batch_size, num_classes)
        if gradient_y.ndim == 1:
            gradient_y = gradient_y.reshape(1, -1)

        batch_size, num_classes = gradient_y.shape

        # Initialize the output gradient
        gradient_x = np.zeros_like(gradient_y)

        for i in range(batch_size):
            # Compute Jacobian for each sample
            jacobian = np.diag(self.y[i]) - np.outer(self.y[i], self.y[i])

            # Compute gradient for each sample
            gradient_x[i] = np.dot(jacobian, gradient_y[i])

        self.y = None
        return gradient_x
