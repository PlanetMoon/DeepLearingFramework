# coding: utf-8

from Activation.Activation import *
import numpy as np


class Tanh(Activation):
    """
    The base class for activation function of neural cell
    """
    def act(self, z):
        a = np.empty_like(z)
        for i in range(len(z)):
            a[i] = 2 / (np.exp(-2 * z[i]) + 1) - 1
        return a

    def prime(self, z):
        p = np.empty_like(z)
        a = self.act(z)
        for i in range(len(z)):
            p[i] = 1 - a[i]**2
        return p
