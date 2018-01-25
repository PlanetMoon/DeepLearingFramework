# coding: utf-8

from Activation.Activation import *
import numpy as np


class ArcTan(Activation):
    """
    The base class for activation function of neural cell
    """
    def act(self, z):
        a = np.empty_like(z)
        for i in range(len(z)):
            a[i] = np.arctan(z[i])
        return a

    def prime(self, z):
        p = np.empty_like(z)
        for i in range(len(z)):
            p[i] = 1 / (1 + z[i]**2)
        return p
