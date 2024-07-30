import numpy as np
from mnist import MNIST

import settings


class Dataset:
    def __init__(self, path, batch_size):
        self.path = path
        self.batch_size = batch_size
        # Load the data
        mndata = MNIST(path)
        images, enc_labels = mndata.load_training()
        images = np.array(images, dtype=settings.default_dtype) / 255.0
        images = (images - np.mean(images, axis=1, keepdims=True)) / np.std(images, axis=1, keepdims=True)

        # Convert labels to one-hot encoding
        num_classes = 10
        one_hot_labels = np.zeros((len(enc_labels), num_classes), dtype=int)
        one_hot_labels[np.arange(len(enc_labels)), enc_labels] = 1

        # shuffle the data
        indices = np.arange(len(images))
        np.random.shuffle(indices)
        images = images[indices]
        one_hot_labels = one_hot_labels[indices]

        self.images = images
        self.labels = one_hot_labels

        self.images = np.linspace(0, 3, 1000).reshape(-1, 1)
        self.labels = np.sin(self.images)
        self.labels = self.labels + np.random.normal(0, .01, self.labels.shape)

        # shuffle
        indices = np.arange(len(self.images))
        np.random.shuffle(indices)
        self.images = self.images[indices]
        self.labels = self.labels[indices]

    def __getitem__(self, item):
        return self.images[item], self.labels[item]

    def __iter__(self):
        for i in range(0, len(self.images), self.batch_size):
            yield self.images[i:i + self.batch_size], self.labels[i:i + self.batch_size]

    def __len__(self):
        return len(self.images) // self.batch_size
