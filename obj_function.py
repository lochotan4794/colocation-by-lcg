import numpy as np


class ObjectiveFunction:

    def __init__(self, L, A, prior, config):
        self.L = L
        self.A = A
        self.prior = prior
        self.mu = config['mu']
        self.lam = config['lambda']

    def _f_(self, x):
        self.tmp = self.L + self.mu*self.A
        t1 = np.dot(x.T, self.tmp)
        t2 = np.dot(t1, x)
        term = np.dot(x.T, np.log(self.prior))
        term = np.array(np.dot(x.T, np.log(self.prior)))
        t3 = np.array(t2 - np.array(self.lam * term))
        return t3[0,0]
    
    def _grad_f(self, x):
        term = np.log(self.prior)
        grad = np.dot(self.tmp, x) - self.lam * term
        return grad
