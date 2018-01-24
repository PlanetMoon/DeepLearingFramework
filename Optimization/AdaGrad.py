# coding: utf-8

import numpy as np
from Optimization.Optimizer import *
from Utils.utils import dot, sqrt


class AdaGrad(Optimizer):
    """
    Adaptive gradient descent
    """
    g_b = dict()
    g_w = dict()
    history_g_b = dict()
    history_g_w = dict()

    def __init__(self, learning_rate, epsilon=1e-7):
        Optimizer.__init__(self, learning_rate)
        self.epsilon = epsilon

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
            if layer.name in self.history_g_b:
                self.history_g_b[layer.name].append(self.g_b[layer.name] / len(mini_batch))
                self.history_g_w[layer.name].append(self.g_w[layer.name] / len(mini_batch))
            else:
                self.history_g_b[layer.name] = [self.g_b[layer.name] / len(mini_batch)]
                self.history_g_w[layer.name] = [self.g_w[layer.name] / len(mini_batch)]
            v_b = self.V(self.history_g_b[layer.name])
            v_w = self.V(self.history_g_w[layer.name])
            eta_w = self.eta(self.learning_rate, self.history_g_w[layer.name][-1], v_w)
            eta_b = self.eta(self.learning_rate, self.history_g_b[layer.name][-1], v_b)
            layer.update_args(- eta_w, - eta_b)

    def g(self, i, layer, w_delta):
        db = w_delta * layer.activation.prime(layer.zs[i])
        self.g_b[layer.name] = self.g_b[layer.name] + db if layer.name in self.g_b else db
        dw = dot(db, layer.xs[i].transpose())
        self.g_w[layer.name] = self.g_w[layer.name] + dw if layer.name in self.g_w else dw
        return dot(layer.weights.transpose(), db)

    def V(self, g_array):
        g_square_array = []
        for i in range(0, len(g_array)):
            g_square_array.append(g_array[i]**2)
        V_square = g_square_array[0]
        for i in range(1, len(g_square_array)):
            V_square += g_square_array[i]
        V = sqrt(V_square)
        return V

    def eta(self, alpha, g, V):
        V += self.epsilon
        return alpha * g / np.sqrt(V)
