# -*- coding: utf-8 -*-

"""
Created by:         Emanuele Bugliarello (@e-bug)
Date created:       7/17/2018
Date last modified: 5/24/2019
"""

from __future__ import absolute_import, division, print_function

import numpy as np


def get_batches(l, n):
    """
    Yield successive n-sized chunks from l.
    Adapted from [https://stackoverflow.com/a/312464].
    """
    for i in range(0, len(l), n):
        yield np.array(l)[i:i+n]

