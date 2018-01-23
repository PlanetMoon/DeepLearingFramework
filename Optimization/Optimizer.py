# coding: utf-8


class Optimizer(object):
    """
    The base class for optimization function
    """
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def execution(self):
        pass
