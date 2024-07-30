from LinearLayer import LinearLayer
from Module import Module
from activations import ReLU


# classify MNist digits: 28x28=784 input, 10 output
class MNistClassifier(Module):
    def __init__(self):
        self.layers = [
            LinearLayer(784, 1568),
            ReLU(),
            LinearLayer(1568, 784),
            ReLU(),
            LinearLayer(784, 392),
            ReLU(),
            LinearLayer(392, 196),
            ReLU(),
            LinearLayer(196, 10),
            ReLU(),
        ]

    def set_parameters(self, dict):
        for layer, layer_dict in dict.items():
            layer.set_parameters(layer)

    def randomize(self):
        [x.randomize() for x in self.layers]

    def backward(self, gradient_y):
        gradient = gradient_y
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)

    def optimize(self, learning_rate):
        [x.optimize(learning_rate) for x in self.layers]

    def get_parameters(self):
        return {i: layer.get_parameters() for i, layer in enumerate(self.layers)}

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x