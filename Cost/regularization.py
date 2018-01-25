# encoding: utf-8

import numpy as np


def L1(lamda, model):
    n = 0.0
    for layer in model.layers:
        n += np.linalg.norm(layer.weights, ord=1)
    return lamda * n


def L2(lamda, model):
    n = 0.0
    for layer in model.layers:
        n += np.linalg.norm(layer.weights)
    return lamda * n / 2
