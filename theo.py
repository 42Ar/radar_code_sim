# -*- coding: utf-8 -*-

import numpy as np
from scipy import linalg


def pad_acf(acf, length):
    if len(acf) >= length:
        return acf[:length]
    r = np.zeros(length)
    r[:len(acf)] = acf
    return r


def fisher(k_i, k_j, Delta_i, Delta_j, c, sigma_inv, M, w):
    eps_i = np.roll(c, k_i)*np.roll(c, k_i - Delta_i)
    eps_j = np.roll(c, k_j)*np.roll(c, k_j - Delta_j)
    eps_LL = np.outer(eps_i, eps_j)
    # square matrix of size len(c), containg diagonal entries
    a = np.sum(eps_LL*sigma_inv[len(c):2*len(c), len(c):2*len(c)]*
               sigma_inv[len(c)+Delta_i:2*len(c)+Delta_i, len(c)+Delta_j:2*len(c)+Delta_j])
    # also square matrix of size len(c), located directly next to the first matrix
    b = np.sum(eps_LL[:,:-M]*sigma_inv[len(c):2*len(c), 2*len(c):3*len(c)-M]*
               sigma_inv[len(c)+Delta_i:2*len(c)+Delta_i, 2*len(c)+Delta_j:3*len(c)+Delta_j - M])
    return (a + 2*b)*w/len(c)/2


def fisher_matrix(c, acfs, N, M, w):
    sigma = np.sum([np.outer(np.roll(np.tile(c, 3), k), np.roll(np.tile(c, 3), k))*linalg.toeplitz(pad_acf(acf, 3*len(c)))
                     for k, acf in enumerate(acfs)], axis=0)
    sigma_inv = linalg.inv(sigma)
    F = np.zeros((N*M, N*M))
    for k1 in range(N):
        for Delta1 in range(1, M+1):
            for k2 in range(N):
                for Delta2 in range(1, M+1):
                    p1 = k1*M + Delta1 - 1
                    p2 = k2*M + Delta2 - 1
                    if p2 >= p1:
                        F[p1, p2] = (fisher(k1, k2, Delta1, Delta2, c, sigma_inv, M, w) +
                                     fisher(k1, k2, Delta1, -Delta2, c, sigma_inv, M, w) +
                                     fisher(k1, k2, -Delta1, Delta2, c, sigma_inv, M, w) +
                                     fisher(k1, k2, -Delta1, -Delta2, c, sigma_inv, M, w))
                    else:
                        F[p1, p2] = F[p2, p1]
    return F


def theo_cov_explicit(h, c, w, Delta1, Delta2, N, acfs):
    assert w%len(c) == 0
    assert len(acfs) == N
    assert Delta1 > 0 and Delta2 > 0
    assert h >= 0 and h < N
    cc = lambda i: c[i%len(c)]
    res = 0
    for k1 in range(N):
        for k2 in range(N):
            for j1 in range(w):
                for j2 in range(w):
                    a = (cc(j1)*cc(j1+Delta1)*cc(j1+h-k2)*cc(j1+Delta1+h-k1)*
                         cc(j2)*cc(j2+Delta2)*cc(j2+h-k2)*cc(j2+Delta2+h-k1))
                    A = a*acfs[k1][abs(j2-j1+Delta2-Delta1)]*acfs[k2][abs(j2-j1)]
                    b = (cc(j1)*cc(j1+Delta1)*cc(j1+h-k2)*cc(j1+Delta1+h-k1)*
                         cc(j2)*cc(j2+Delta2)*cc(j2+h-k1)*cc(j2+Delta2+h-k2))
                    B = b*acfs[k1][abs(j1-j2+Delta1)]*acfs[k2][abs(j2-j1+Delta2)]
                    res += A + B
    return res/(2*w**2)


def theo_cov(h, c, w, Delta1, Delta2, N, acfs):
    assert w%len(c) == 0
    assert len(acfs) == N
    assert Delta1 > 0 and Delta2 > 0
    assert h >= 0 and h < N
    assert all(len(acf) >= w + max(Delta1, Delta2) for acf in acfs), f"where w={w}"
    c = np.array(c)
    def eps_hat(Delta, l1, l2):
        return np.tile(c*np.roll(c, -Delta)*np.roll(c, -h + l2)*np.roll(c, -Delta - h + l1), w//len(c))
    def shifted_toeplitz(k, shift):
        if shift < 0:
            return linalg.toeplitz(acfs[k][:w - shift])[:w, -shift:]
        else:
            return linalg.toeplitz(acfs[k][:w + shift])[shift:, :w]
    res = 0
    for k1 in range(N):
        for k2 in range(N):
            eps_hat1 = eps_hat(Delta1, k1, k2)
            eps_hat2 = eps_hat(Delta2, k1, k2)
            eps_hat3 = eps_hat(Delta2, k2, k1)
            R1 = shifted_toeplitz(k1, Delta1 - Delta2)
            R2 = shifted_toeplitz(k2, 0)
            R3 = shifted_toeplitz(k1, Delta1)
            R4 = shifted_toeplitz(k2, -Delta2)
            A = np.sum(np.outer(eps_hat1, eps_hat2)*R1*R2)
            B = np.sum(np.outer(eps_hat1, eps_hat3)*R3*R4)
            res += A + B
    return res/(2*w**2)


def theo_cov_with_cutoff(h, c, w, Delta1, Delta2, cutoff, N, acfs):
    assert cutoff > 0
    segment_size = len(c)*((cutoff + max(Delta1, Delta2) + 2*len(c))//len(c))
    if w > segment_size:
        seg_val = segment_size**2*theo_cov(h, c, segment_size, Delta1, Delta2, N, acfs)
        sub_seg_val = (segment_size - len(c))**2*theo_cov(h, c, segment_size - len(c), Delta1, Delta2, N, acfs)
        return ((seg_val - sub_seg_val)*((w - segment_size)//len(c) + 1) + sub_seg_val)/w**2
    else:
        return theo_cov(h, c, w, Delta1, Delta2, N, acfs)


def theo_var_approx(h, w, Delta, acfs, cutoff):
    A = sum(acf[0] for acf in acfs)**2/(2*w)
    B = sum(acf[Delta]**2 for acf in acfs)/(2*w)
    C = sum((w - k)*(acfs[h][k]**2 + acfs[h][k + Delta]*acfs[h][np.abs(-k + Delta)]) for k in range(1, cutoff))/w**2
    return A + B + C


def theo_cov_matrix_with_cutoff(h, c, w, cutoff, N, acfs, M):
    res = np.zeros((M, M))
    for i in range(M):
        for j in range(i, M):
            res[i, j] = theo_cov_with_cutoff(h, c, w, i + 1, j + 1, cutoff, N, acfs)
            res[j, i] = res[i, j]
    return res