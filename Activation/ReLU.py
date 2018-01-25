# coding: utf-8

from Activation.Activation import *
import numpy as np


class ReLU(Activation):
    """
    The base class for activation function of neural cell
    """
    def act(self, z):
        a = np.empty_like(z)
        for i in range(len(z)):
            a[i] = np.max(0, z[i])
        return a

    def prime(self, z):
        p = np.empty_like(z)
        for i in range(len(z)):
            p[i] = 0 if z[i] < 0 else 1
        return p
