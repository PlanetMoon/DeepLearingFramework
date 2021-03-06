# encoding: utf-8

from Activation.Activation import *
import numpy as np


class Sigmoid(Activation):
    """
    Sigmoid: 1 / (1 + exp(-z))
    """
    def act(self, z):
        a = np.empty_like(z)
        for i in range(len(z)):
            a[i] = 1.0 / (1.0 + np.exp(-z[i]))
        return a

    def prime(self, z):
        p = np.empty_like(z)
        a = self.act(z)
        for i in range(len(a)):
            p[i] = a[i] * (1 - a[i])
        return p
