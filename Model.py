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

    def back_prop(self, i, x, y):
        nabla_b = dict()
        nabla_w = dict()
        a = self.feed_forward(i, x)
        c = self.cost(a, y)
        self.layers.reverse()
        w_delta = 1 * c
        for layer in self.layers:
            (db, dw, wd) = layer.back_prop(i, w_delta)
            nabla_b[layer.name] = db
            nabla_w[layer.name] = dw
            w_delta = wd
        self.layers.reverse()
        return nabla_b, nabla_w

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
