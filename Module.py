from abc import ABC, abstractmethod


class Module(ABC):
    @abstractmethod
    def forward(self, input):
        pass

    def get_parameters(self):
        return {}

    def set_parameters(self, dict):
        pass

    def randomize(self):
        pass

    @abstractmethod
    def backward(self, gradient_y):
        pass

    @abstractmethod
    def optimize(self, learning_rate):
        pass
