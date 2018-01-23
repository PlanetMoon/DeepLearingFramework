# encoding: utf-8

from Activation import Activation
import numpy as np


class Sigmoid(Activation):
    """
    Sigmoid: 1 / (1 + exp(-z))
    """
    def __init__(self):
        Activation.__init__(self)
        pass

    def act(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def prime(self, z):
        return self.act(z) * (1 - self.act(z))
