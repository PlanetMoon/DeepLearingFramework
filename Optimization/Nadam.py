# coding: utf-8

import numpy as np
from Optimization.Optimizer import *
from Utils.utils import dot, sqrt


class Adam(Optimizer):
    """
    Adaptive + Momentum + Nesterov
    """
    g_b = dict()
    g_w = dict()
    v_b = dict()
    v_w = dict()
    m_b = dict()
    m_w = dict()
    eta_w = dict()
    eta_b = dict()

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
            for layer in model.layers:
                if layer.name in self.eta_w:
                    layer.update_args(- self.eta_w[layer.name], - self.eta_b[layer.name])
            a = model.feed_forward(i, x)
            c = model.cost(a, y)
            model.layers.reverse()
            w_delta = 1 * c
            for layer in model.layers:
                w_delta = self.g(i, layer, w_delta)
            model.layers.reverse()
            for layer in model.layers:
                if layer.name in self.eta_w:
                    layer.update_args(self.eta_w[layer.name], self.eta_b[layer.name])
            i += 1
        self.m(len(mini_batch))
        for layer in model.layers:
            self.v_b[layer.name] = self.V(self.v_b[layer.name], self.g_b[layer.name] / len(mini_batch))
            self.v_w[layer.name] = self.V(self.v_w[layer.name], self.g_w[layer.name] / len(mini_batch))
            self.eta_w[layer.name] = self.eta(self.learning_rate, self.m_w[layer.name][-1], self.v_w[layer.name])
            self.eta_b[layer.name] = self.eta(self.learning_rate, self.m_b[layer.name][-1], self.v_b[layer.name])
            layer.update_args(- eta_w, - eta_b)

    def g(self, i, layer, w_delta):
        db = w_delta * layer.activation.prime(layer.zs[i])
        self.g_b[layer.name] = self.g_b[layer.name] + db if layer.name in self.g_b else db
        dw = dot(db, layer.xs[i].transpose())
        self.g_w[layer.name] = self.g_w[layer.name] + dw if layer.name in self.g_w else dw
        return dot(layer.weights.transpose(), db)

    def V(self, v, g_array):
        V_square = g_array[-1]**2
        V = self.beta2 * v + (1 - self.beta2) * sqrt(V_square)
        return V

    def m(self, len):
        for (layer_name, g_w) in self.g_w.items():
            delta_w = (1 - self.beta1) * g_w / len
            self.m_w[layer_name] = delta_w + self.beta1 * self.m_w[layer_name] if layer_name in self.m_w else delta_w
        for (layer_name, g_b) in self.g_b.items():
            delta_b = (1 - self.beta1) * g_b / len
            self.m_b[layer_name] = delta_b + self.beta1 * self.m_b[layer_name] if layer_name in self.m_b else delta_b

    def eta(self, alpha, g, V):
        V += self.epsilon
        return alpha * g / np.sqrt(V)
