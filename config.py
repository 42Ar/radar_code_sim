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
w = 30*len(c)
off = len(c)
n = off + w + len(c)
N = 7
M = 5
cc = np.tile(c, math.ceil(n/len(c)))[:n]
samples = 1000
method="hybrid"
K = 10  # where to cutoff the ACF when mode="hybrid"

def check_code(k, h, Delta):
    L = len(c)
    cc = c*3
    r = sum((cc[L + j + Delta]*cc[L + j]*
             cc[L + j + Delta + h - k]*cc[L + j + h - k])
            for j in range(L))
    if r != 0:
        print("code error:", k, h, Delta)

for k in range(N):
    for h in range(N):
        if k != h:
            for Delta in range(1, M+1):
                check_code(k, h, Delta)