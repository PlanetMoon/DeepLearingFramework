# coding: utf-8
from Optimization.Optimizer import *


class SGD(Optimizer):
    """
    Stochastic gradient descent
    """
    def execution(self, model, mini_batch):
        nabla_b = dict()
        nabla_w = dict()
        i = 0
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = model.back_prop(i, x, y)
            for (layer_name, db) in delta_nabla_b.items():
                nabla_b[layer_name] = nabla_b[layer_name] + db if layer_name in nabla_b else db
            for (layer_name, dw) in delta_nabla_w.items():
                nabla_w[layer_name] = nabla_w[layer_name] + dw if layer_name in nabla_w else dw
            i += 1
        for layer in model.layers:
            layer.update_args(- self.learning_rate * nabla_w[layer.name] / len(mini_batch), - self.learning_rate * nabla_b[layer.name] / len(mini_batch))
