# -*- coding: utf-8 -*-

"""
Created by:         Emanuele Bugliarello (@e-bug)
Date created:       7/17/2018
Date last modified: 5/24/2019
"""

from __future__ import absolute_import, division, print_function

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
        X: Matrix of true values. Missing entries are set to zeros.
        P: Matrix of predictions. Missing entries are set to zeros.
        O: Boolean matrix of entries to take into account for RMSE computation. Defaults to `1` (all entries)
    Returns:
        RMSE value
    """
    return np.sqrt(np.sum(np.power(O * (X - P), 2)) / np.count_nonzero(O))


def mae(X, P, O=1):
    """
    Compute MAE between values in matrix X and matrix P at the indices specified by the boolean matrix O.
    
    Args:
        X: Matrix of true values. Missing entries are set to zeros.
        P: Matrix of predictions. Missing entries are set to zeros.
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


def precision_at_n(n, X, P, O=1):
    """
    Compute Precision@N between values in matrix X and matrix P at the indices specified by the boolean matrix O.
    
    Args:
        X: Matrix of true values. Missing entries are set to zeros.
        P: Matrix of predictions. Missing entries are set to zeros.
        O: Boolean matrix of entries to take into account for Precision@N computation. Defaults to `1` (all entries)
    Returns:
        Precision@N value
    """
    P[O == 0] = -1
    precisions = []
    for c in range(X.shape[1]):
        y_true = set(np.argsort(X[:, c])[::-1][:n])
        y_pred = set(np.argsort(P[:, c])[::-1][:n])
        tp = len(y_pred.intersection(y_true))
        precision = tp / n
        precisions.append(precision)
    return np.mean(precisions)


def recall_at_n(n, X, P, O=1):
    """
    Compute Recall@N between values in matrix X and matrix P at the indices specified by the boolean matrix O.
    
    Args:
        X: Matrix of true values. Missing entries are set to zeros.
        P: Matrix of predictions. Missing entries are set to zeros.
        O: Boolean matrix of entries to take into account for Recall@N computation. Defaults to `1` (all entries)
    Returns:
        Recall@N value
    """
    P[O == 0] = -1
    threshold = -1
    recalls = []
    for c in range(X.shape[1]):
        y_true = set(np.argsort(X[:, c])[::-1][:n])
        y_pred = set(np.argsort(P[:, c])[::-1][:n])
        tp = len(y_pred.intersection(y_true))
        den = np.sum(X[:, c] > threshold)
        recall = tp / den
        recalls.append(recall)
    return np.mean(recalls)

