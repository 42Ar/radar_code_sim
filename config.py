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
import code_utils

code = "5_7"
code = "6_8_golden"
N = 7
M = 5
if code == "5_7":
    c = [-1, +1, +1, -1, +1, -1, +1, +1, -1, -1, +1, +1, -1, -1, -1, +1, -1, +1, -1, +1, +1, -1, +1, -1, +1, +1, +1, +1, +1, -1, +1, +1, +1, -1, -1, -1, +1, +1, +1, +1]
    w = 30*len(c)
    off = len(c)
    samples = 1000
elif code == "5_7_48":
    c = [-1, +1, -1, -1, +1, +1, +1, +1, +1, +1, +1, +1, +1, -1, +1, +1, +1, +1, -1, +1, +1, -1, -1, -1, -1, +1, -1, -1, +1, +1, +1, -1, +1, +1, -1, +1, +1, -1, -1, +1, +1, +1, +1, -1, -1, -1, -1, +1]
    w = 30*len(c)
    off = len(c)
    samples = 1000*40//48
elif code == "6_8_golden":
    N = 8
    M = 6
    c = [-1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, 1, -1, -1, -1, -1, 1, -1, 1, 1, -1, -1, -1, 1, 1, -1, -1, -1, -1, 1, -1, -1, -1, 1, -1, -1, 1, 1, -1, 1, -1, -1, 1, -1, -1, 1]
    w = 30*len(c)
    off = len(c)
    samples = 1000
elif code == "random":
    w = 500
    c = list((np.random.default_rng(468).uniform(size=w)>0.5)*2 - 1)
    off = 20
    samples = 1000
else:
    raise ValueError("unknown code")
n = off + w + off
cc = np.tile(c, math.ceil(n/len(c)))[:n]
method="hybrid"
max_fisher_size = 500
K = 10  # where to cutoff the ACF when mode="hybrid"

if code != "random":
    code_utils.full_code_check(c, N, M)
