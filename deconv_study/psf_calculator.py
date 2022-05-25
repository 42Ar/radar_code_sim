# -*- coding: utf-8 -*-

import numpy as np


def calc_PSF(q, g):
    assert abs(q) <= 1
    S = len(g)
    res = np.zeros((2*S-1, 2*S-1))
    for s1 in range(S):
        for s2 in range(S):
            for s3 in range(S):
                for s4 in range(S):
                    v = g[s1]*g[s2]*g[s3]*g[s4]
                    if s2 - s1 == s4 - s3 + q:
                        delta_index = S - 1 + s2 - s4
                        H_index = S - 1 + s2 - s1
                        assert delta_index >= 0 and H_index >= 0
                        res[delta_index, H_index] += v
    return res


def run_tests():
    test = [-1, 1]
    o = calc_PSF(0, test)
    assert np.all(o == np.array([[0., 1., 0.], [1., 2., 1.], [0., 1., 0.]]))
    o = calc_PSF(-1, test)
    assert np.all(o == np.array([[-1., -1.,  0.], [-1., -1.,  0.], [ 0.,  0.,  0.]]))
    o = calc_PSF(1, test)
    assert np.all(o == np.array([[ 0.,  0.,  0.], [ 0., -1., -1.], [ 0., -1., -1.]]))


run_tests()