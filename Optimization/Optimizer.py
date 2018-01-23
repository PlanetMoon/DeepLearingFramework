# coding: utf-8


class Optimizer(object):
    """
    The base class for optimization function
    """
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def execution(self):
        pass

    def g(self):
        """
        Compute the gradient
        """
        pass

    def m(self):
        """
        Compute the first-order momentum
        """
        pass

    def V(self):
        """
        Compute the Second-order momentum
        """
        pass

    def eta(self):
        """
        Compute the current descent gradient
        """
        pass
