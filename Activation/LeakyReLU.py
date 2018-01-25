# coding: utf-8

from Activation.Activation import *
import numpy as np


class LeakyReLU(Activation):
    """
    The base class for activation function of neural cell
    """
    def __init__(self, alpha):
        self.alpha = alpha

    def act(self, z):
        a = np.empty_like(z)
        for i in range(len(z)):
            a[i] = self.alpha * z[i] if z[i] < 0 else z[i]
        return a

    def prime(self, z):
        p = np.empty_like(z)
        for i in range(len(z)):
            p[i] = self.alpha if z[i] < 0 else 1
        return p
