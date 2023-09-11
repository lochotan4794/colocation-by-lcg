import numpy as np
from LPCG_optimizer import *

class Optimizer:


    def __init__(self, obj, config):
        self.ObjFunc = obj
        self.config = config
        self.M = self.config['M']
        self.nb = self. M * self.config['Number_of_images']


    def RUN(self):
        X0 = np.random.choice([0, 1], size=(self.nb, 1), p=[1./2, 1./2])
        if self.config['algo'] == 'LCPG':
            solution, logs = LPCG_Optimizer(X0, self.ObjFunc._f_, self.ObjFunc._grad_f, self.config['max_iter'])
        return solution, logs