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
#                                             Learning Rate Decay functions                                            #
#                                                                                                                      #
# ==================================================================================================================== #

def inverse_time_decay(lr0, step, decay_rate, decay_steps, staircase=False):
    """
    Inverse time decay to the initial learning rate `lr0`:
        decayed_learning_rate = lr0 / (1 + decay_rate * step / decay_steps)
    If the argument `staircase` is `True`, then `step / decay_steps` is an integer division 
    and the decayed learning rate follows a staircase function.
    
    Args:
        lr0: Initial learning rate
        step: Iteration step to use for the decay computation
        decay_rate: Decay rate
        decay_steps: Number of iteration steps over which to decay the learning rate
        staircase: Boolean, if `True` decay the learning rate at discrete intervals. Defaults to `False`
    Returns:
        decayed learning rate at iteration step `step`
    """
    ratio = step / decay_steps
    if staircase:
        ratio = np.floor(ratio)
    return lr0 / (1 + decay_rate * ratio)


def exponential_decay(lr0, step, decay_rate, decay_steps, staircase=False):
    """
    Exponential decay to the initial learning rate `lr0`:
        decayed_learning_rate = lr0 * decay_rate ^ (step / decay_steps)
    If the argument `staircase` is `True`, then `step / decay_steps` is an integer division 
    and the decayed learning rate follows a staircase function.    
    
    Args:
        lr0: Initial learning rate
        step: Iteration step to use for the decay computation
        decay_rate: Decay rate
        decay_steps: Number of iteration steps over which to decay the learning rate
        staircase: Boolean, if `True` decay the learning rate at discrete intervals. Defaults to `False`
    Returns:
        decayed learning rate at iteration step `step`
    """
    ratio = step / decay_steps
    if staircase:
        ratio = np.floor(ratio)
    return lr0 * np.power(decay_rate, ratio)


def natural_exp_decay(lr0, step, decay_rate, decay_steps, staircase=False):
    """
    Natural exponential decay to the initial learning rate `lr0`:
        decayed_learning_rate = lr0 * exp(-decay_rate * step)
    If the argument `staircase` is `True`, then `step / decay_steps` is an integer division 
    and the decayed learning rate follows a staircase function.
    
    Args:
        lr0: Initial learning rate
        step: Iteration step to use for the decay computation
        decay_rate: Decay rate
        decay_steps: Number of iteration steps over which to decay the learning rate
        staircase: Boolean, if `True` decay the learning rate at discrete intervals. Defaults to `False`
    Returns:
        decayed learning rate at iteration step `step`

    """
    ratio = step / decay_steps
    if staircase:
        ratio = np.floor(ratio)
    return lr0 * np.exp(-decay_rate * ratio)


def piecewise_constant(lr0, step, iter_boundaries, lr_values):
    """
    Piecewise constant from boundaries and interval values.
        Example: use a learning rate that's `lr0` for the first 1000 steps, 
                 0.5 for the next 1000 steps, and 0.1 for any additional steps.
    
    Args:
        lr0: Initial learning rate
        step: Iteration step to use for the decay computation
        iter_boundaries: List of iteration steps at which to change the learning rate
        lr_values: List of learning rate values to be set starting from the corresponding step in `iter_boundaries`
    Returns:
        decayed learning rate at iteration step `step`
    """
    iter_boundaries = np.array([0] + iter_boundaries)
    lr_values = np.array([lr0] + lr_values)
    idx = np.argwhere(iter_boundaries <= step)[-1, 0]
    return lr_values[idx]


def polynomial_decay(lr0, step, decay_steps, lr_end=0.0001, power=1.0):
    """
    Polynomial decay to the initial learning rate `lr0` to reach a final learning rate `lr_end` in `decay_steps` steps:
        step = min(step, decay_steps)
        decayed_learning_rate = (lr0 - lr_end) * (1 - step / decay_steps) ^ (power) + lr_end
    
    Args:
        lr0: Initial learning rate
        step: Iteration step to use for the decay computation
        decay_steps: Number of iteration steps over which to decay the learning rate
        lr_end: Final learning rate. Defaults to `0.0001`
        power: Power of the polynomial. Defaults to linear (1.0)
    Returns:
        decayed learning rate at iteration step `step`
    """
    step = np.minimum(step, decay_steps)
    ratio = step / decay_steps
    return (lr0 - lr_end) * np.power(1 - ratio, power) + lr_end

