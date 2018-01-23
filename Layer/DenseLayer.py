# coding: utf-8

import numpy as np


class DenseLayer(object):
    """
    The base class of Neural Layer
    """

    def __init__(self, input_num, size, activation, name):
        self.input_num = input_num
        self.size = size
        self.name = name
        self.weights = np.random.randn(size, input_num)
        self.bias = np.random.randn(size, 1)
        self.activation = activation
        self.xs = dict()
        self.zs = dict()

    def dot(self, x, y):
        return np.dot(x, y)

    def feed_forward(self, batch_i, x):
        self.xs[batch_i] = x
        z = self.dot(self.weights, x) + self.bias
        self.zs[batch_i] = z
        return self.activation.act(z)

    def back_prop(self, batch_i, w_delta):
        d = w_delta * self.activation.prime(self.zs[batch_i])
        dw = self.dot(d, self.xs[batch_i].transpose())
        wd = self.dot(self.weights.transpose(), d)
        return d, dw, wd

    def update_args(self, dw, db):
        self.weights += dw
        self.bias += db

    def __str__(self):
        return "{0} = input: {1}\t size: {2}\n".format(self.name, self.input_num, self.size)
