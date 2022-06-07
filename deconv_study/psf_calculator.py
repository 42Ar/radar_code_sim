# -*- coding: utf-8 -*-

import numpy as np


def calc_delta_min(q, S):
    return max(q, 0) - S + 1


def calc_delta_max(q, S):
    return min(q, 0) + S - 1


def calc_PSF(q, g):
    S = len(g)
    delta_min = calc_delta_min(q, S)
    delta_max = calc_delta_max(q, S)
    delta_size = delta_max - delta_min + 1
    res = np.zeros((delta_size, 2*S - 1))
    for s1 in range(S):
        for s2 in range(S):
            for s3 in range(S):
                for s4 in range(S):
                    v = g[s1]*g[s2]*g[s3]*g[s4]
                    if s2 - s4 == s1 - s3 + q:
                        delta = s2 - s4
                        delta_index = delta - delta_min
                        H = S - 1 - s1 + s2 
                        assert delta_index >= 0 and H >= 0
                        res[delta_index, H] += v
    return res


def calc_PSF_array(g):
    return [calc_PSF(q, g) for q in range(2*len(g) - 1)]


def get_PSF(psf_array, q):
    if q >= 0:
        return psf_array[q]
    else:
        return np.flip(psf_array[-q], 1)


def run_tests():
    test = [-1, 1]
    o = calc_PSF(0, test)
    assert np.all(o == np.array([[0., 1., 0.], [1., 2., 1.], [0., 1., 0.]]))
    o = calc_PSF(-1, test)
    assert np.all(o == np.array([[-1., -1.,  0.], [-1., -1.,  0.]]))
    o = calc_PSF(1, test)
    assert np.all(o == np.array([[ 0., -1., -1.], [ 0., -1., -1.]]))


run_tests()