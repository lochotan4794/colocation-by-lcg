import numpy as np
import time
import logging
import os
from tabulate import tabulate
import signal

from . import utils
from . import globs


class Algorithm:
    def __init__(self, feasible_region, objectiveFunction, run_config):
        # parameter setting
        self.feasible_region = feasible_region

    def fwStep_update_alpha_S(self, step_size, index_s, s):
        if step_size == 1:
            self.s_list = [s]  # set the s list to be this atom only
            self.alpha = np.array([step_size])  # the weight for this atom is 1
        else:
            self.alpha = (1 - step_size) * self.alpha  # update weight for all the atoms originally in the s_list
            if index_s is None:  # s is new
                self.s_list.append(s)
                self.alpha = np.append(self.alpha, step_size)
            else:  # old atom
                self.alpha[index_s] += step_size

    def run_algorithm(self):
        return 0













