import numpy as np

from Module import Module


class CrossEntropyLoss(Module):
    def optimize(self, learning_rate):
        pass

    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        return -np.sum(y_true * np.log(y_pred + 1e-8))

    def backward(self):
        return self.y_pred - self.y_true
