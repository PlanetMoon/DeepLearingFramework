# encoding: utf-8

import numpy as np


def cost_linear(output_activations, y):
    return y - output_activations


def cost_square(output_activations, y):
    return (y - output_activations)**2


def cost_exp(output_activations, y):
    return np.exp(-y * output_activations)


def cost_log(output_activations, y):
    return np.log2(1 + np.exp(-y * output_activations))
