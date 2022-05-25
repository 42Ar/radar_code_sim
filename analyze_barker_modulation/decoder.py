# -*- coding: utf-8 -*-

import numpy as np
import scipy.linalg as linalg

def decode(m, off, w, c, N, M):
    assert m.ndim == 2
    assert w % len(c) == 0  # this must hold for the cancellation property to be fullfilled
    cc = np.tile(c, m.shape[0]//len(c) + 1)
    res = np.zeros((N, M, m.shape[1]), dtype=np.complex128)
    for h in range(N):
        d1 = m[off + h:off + h + w]*cc[off:off + w, np.newaxis]
        for Delta in range(1, M + 1):
            d2 = m[off + h + Delta:off + h + Delta + w]*cc[off + Delta:off + Delta + w, np.newaxis]
            res[h, Delta - 1] += np.mean(np.conj(d2)*d1, axis=0)
    return res


def calc_approx_deconv_matrices(g, W, trunc = 0, use_svd=True):
    K = np.zeros((W, len(g) + W))
    gp = np.flip(g)  # gamma'
    for w in range(W):
        K[w, w] += 1
        K[w, w + 1] += 1
        for s in range(1, len(g)):
            K[w, w + s] += 1 + 2*gp[s - 1]*gp[s]
            K[w, w + s + 1] += 1
    if use_svd:
        u, s, v = linalg.svd(K)
        if trunc is None:
            trunc = W
        v_trunc = v[:trunc, :]
        X = v_trunc.T @ np.diag(1/s[:trunc]**2) @ v_trunc
        return X, K, s
    else:
        inv = linalg.inv(K.T@K)
        return inv, K, None
        
def deconvolute_matrix_method_approx(d, X, K):
    M = X @ K.T
    res = np.empty((d.shape[0], d.shape[1], X.shape[0]), dtype=np.complex128)
    for i in range(d.shape[0]):
        for j in range(d.shape[1]):
            res[i, j] = M @ d[i, j]
    return res

