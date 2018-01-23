# coding: utf-8

import numpy as np
from Utils.utils import dot


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

    def feed_forward(self, batch_i, x):
        self.xs[batch_i] = x
        z = dot(self.weights, x) + self.bias
        self.zs[batch_i] = z
        return self.activation.act(z)

    def update_args(self, dw, db):
        self.weights += dw
        self.bias += db

    def __str__(self):
        return "{0} = input: {1}\t size: {2}\n".format(self.name, self.input_num, self.size)
