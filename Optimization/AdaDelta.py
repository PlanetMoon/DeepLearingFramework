# coding: utf-8

import numpy as np
from Optimization.Optimizer import *
from Utils.utils import dot, sqrt


class AdaDelta(Optimizer):
    """
    Adaptive gradient descent with delta window
    """
    g_b = dict()
    g_w = dict()
    v_b = dict()
    v_w = dict()

    def __init__(self, learning_rate, beta2=0.999, epsilon=1e-6):
        Optimizer.__init__(self, learning_rate)
        self.epsilon = epsilon
        self.beta2 = beta2

    def reset(self):
        self.g_w = dict()
        self.g_b = dict()

    def execution(self, model, mini_batch):
        i = 0
        self.reset()
        for x, y in mini_batch:
            a = model.feed_forward(i, x)
            c = model.cost(a, y)
            model.layers.reverse()
            w_delta = 1 * c
            for layer in model.layers:
                w_delta = self.g(i, layer, w_delta)
            model.layers.reverse()
            i += 1
        for layer in model.layers:
            self.v_b[layer.name] = self.V(self.v_b[layer.name], self.g_b[layer.name] / len(mini_batch))
            self.v_w[layer.name] = self.V(self.v_w[layer.name], self.g_w[layer.name] / len(mini_batch))
            eta_w = self.eta(self.learning_rate, self.g_w[layer.name], self.v_w[layer.name])
            eta_b = self.eta(self.learning_rate, self.g_b[layer.name], self.v_b[layer.name])
            layer.update_args(- eta_w, - eta_b)

    def g(self, i, layer, w_delta):
        db = w_delta * layer.activation.prime(layer.zs[i])
        self.g_b[layer.name] = self.g_b[layer.name] + db if layer.name in self.g_b else db
        dw = dot(db, layer.xs[i].transpose())
        self.g_w[layer.name] = self.g_w[layer.name] + dw if layer.name in self.g_w else dw
        return dot(layer.weights.transpose(), db)

    def V(self, v, g):
        V_square = g**2
        V = self.beta2 * v + (1 - self.beta2) * sqrt(V_square)
        return V

    def eta(self, alpha, g, V):
        V += self.epsilon
        return alpha * g / np.sqrt(V)
