import numpy as np
import itertools

from scipy.special import gamma
import CO_data

class multifield(object):
    def __init__(self, z, line_list):

        self._check_assigned_bias_(line_list)
    
    def _check_assigned_bias_(self, line_list):
        for line in line_list:
            assert line.bias_assigned, "Bias not assigned for line "+line.key
