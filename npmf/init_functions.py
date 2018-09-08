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
#                                               Initialization functions                                               #
#                                                                                                                      #
# ==================================================================================================================== #

def rand_init(num_users, num_items, num_features, rand_fn=np.random.rand):
    """
    Randomly initialize the factor matrices.
    
    Args:
        num_users: Number of rows in the ratings matrix
        num_items: Number of columns in the ratings matrix
        num_features: Number of latent factors in the generated factor matrices
        rand_fn: Random values generator function. Defaults to np.random.rand (uniformly in [0,1))
    Returns:
        Factor matrix of users (num_users, num_features), factor matrix of items (num_items, num_features) 
    """

    user_features = rand_fn(num_users, num_features)
    item_features = rand_fn(num_items, num_features)

    return user_features, item_features
