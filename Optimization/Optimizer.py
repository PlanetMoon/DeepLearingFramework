# coding: utf-8


class Optimizer(object):
    """
    The base class for optimization function
    Reference: https://zhuanlan.zhihu.com/p/32230623
    Optimization can be divided to four steps:
    1. Calculate the gradient for current function: g_t = Laplace(f(w_t))
    2. Calculate the first-order momemtum and second-order momemtum: m_t = Phi(g_1, g_2, ... , g_t); V_t = Psi(g_1, g_2, ... , g_t)
    3. Calculate the gradient descent: delta_t = alpha * m_t / sqrt(V_t)
    4. Update the function: w_(t+1) = w_t - delta_t
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
