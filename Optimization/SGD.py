# coding: utf-8

from Optimization.Optimizer import *
from Utils.utils import dot


class SGD(Optimizer):
    """
    Stochastic gradient descent
    """
    nabla_b = dict()
    nabla_w = dict()
    
    def reset(self):
        nabla_w = dict()
        nabla_b = dict()

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
            eta_w = self.eta(self.learning_rate, self.nabla_w[layer.name])
            eta_b = self.eta(self.learning_rate, self.nabla_b[layer.name])
            layer.update_args(- eta_w / len(mini_batch), - eta_b / len(mini_batch))

    def g(self, i, layer, w_delta):
        db = w_delta * layer.activation.prime(layer.zs[i])
        self.nabla_b[layer.name] = self.nabla_b[layer.name] + db if layer.name in self.nabla_b else db
        dw = dot(db, layer.xs[i].transpose())
        self.nabla_w[layer.name] = self.nabla_w[layer.name] + dw if layer.name in self.nabla_w else dw
        return dot(layer.weights.transpose(), db)

    def eta(self, alpha, g):
        return alpha * g
