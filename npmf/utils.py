# -*- coding: utf-8 -*-
"""
Created by e-bug on 8/31/18
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def get_batches(l, n):
    """
    Yield successive n-sized chunks from l.
    Adapted from [https://stackoverflow.com/a/312464].
    """
    for i in range(0, len(l), n):
        yield np.array(l)[i:i+n]
