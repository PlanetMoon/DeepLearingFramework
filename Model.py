# coding: utf-8

import numpy as np


class Model(object):
    """
    The base class of the Neural Network
    """

    def __init__(self, optimization, cost):
        self.layers = []
        self.cost = cost
        self.optimization = optimization

    def append(self, layer):
        self.layers.append(layer)

    def feed_forward(self, i, x):
        a = x
        for layer in self.layers:
            a = layer.feed_forward(i, a)
        return a

    def train_on_batch(self, mini_batch):
        self.optimization.execution(self, mini_batch)

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feed_forward(0, x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def predict(self, data):
        value = self.feed_forward(0, data)
        return value.tolist().index(max(value))

    def save(self):
        pass  # save the network

    def load(self):
        pass
