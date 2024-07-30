import numpy as np

from Module import Module


class MSE(Module):
    def optimize(self, learning_rate):
        pass

    def forward(self, y_pred, y_true):
        assert y_pred.shape[-1] == y_true.shape[-1]

        self.y_pred = y_pred
        self.y_true = y_true
        return np.mean((y_pred - y_true) ** 2) * 0.5
    def backward(self):
        return self.y_pred - self.y_true


class CrossEntropyLoss(Module):
    def optimize(self, learning_rate):
        pass

    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        return -np.sum(y_true * np.log(y_pred + 1e-8))

    def backward(self):
        return self.y_pred - self.y_true