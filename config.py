#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
        #   #   #
       / \ / \ / \
      #   #   #   \
     / \ / \ / \   \
    #   #   #   \   \
   / \ / \ / \   \   \

"""
import numpy as np
import math

# M = 5, N = 7
c = [-1, +1, -1, -1, +1, +1, +1, +1, +1, +1, +1, +1, +1, -1, +1, +1, -1, -1, +1, -1, +1, +1, -1, -1, -1, +1, +1, -1, +1, +1, +1, -1, +1, -1, -1, -1, -1, +1, +1, +1]
n = len(c)*3
N = 7
M = 5
cc = np.tile(c, math.ceil(n/len(c)))[:n]
rng = np.random.default_rng(42)
samples = 10000
method="hybrid"
K = 10  # where to cutoff the ACF when mode="hybrid"
