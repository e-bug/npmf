# -*- coding: utf-8 -*-
"""
Created by e-bug on 7/17/18
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


# ==================================================================================================================== #
#                                                                                                                      #
#                                                     Error metrics                                                    #
#                                                                                                                      #
# ==================================================================================================================== #

def rmse(X, P, O=1):
    """
    Compute RMSE between values in matrix X and matrix P at the indices specified by the boolean matrix O.
    
    Args:
        X: Matrix of true values
        P: Matrix of predictions
        O: Boolean matrix of entries to take into account for RMSE computation. Defaults to `1` (all entries)
    Returns:
        RMSE value
    """
    return np.sqrt(np.sum(np.power(O * (X - P), 2)) / np.count_nonzero(O))


def mae(X, P, O=1):
    """
    Compute MAE between values in matrix X and matrix P at the indices specified by the boolean matrix O.
    
    Args:
        X: Matrix of true values
        P: Matrix of predictions
        O: Boolean matrix of entries to take into account for MAE computation. Defaults to `1` (all entries)
    Returns:
        MAE value
    """
    return np.sum(np.abs(O * (X - P))) / np.count_nonzero(O)


def se(errors):
    """
    Compute Standard Error (SE) of a sequence of values.
    
    Args:
        errors: Array of values of which to compute SE
    Returns:
        SE value
    """
    return np.std(errors) / np.sqrt(len(errors))
