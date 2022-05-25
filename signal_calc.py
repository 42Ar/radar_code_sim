# -*- coding: utf-8 -*-

import numpy as np
from scipy.linalg import sqrtm, toeplitz
from scipy.fftpack import dct, fft
import matplotlib.pyplot as plt
import scipy.stats


def gen_layer_ACF(t, h, N, mode):
    if mode == "spread":
        width = 3*h/N + 1
        return (h/N + 1)*np.exp(-t/width)
    elif mode == "strong_upper_reflectivity":
        amp = 0.02 if h == 0 else 1
        return amp*np.exp(-t)
    elif mode == "strong_upper_reflectivity2":
        amp = 0.2 if h == 0 else 1
        width = 1 if h == 0 else 10
        return amp*np.exp(-t/width)
    elif mode == "strong_upper_reflectivity_zero_corr":
        if h == 0:
            return np.exp(-t)
        else:
            ACF = np.zeros(len(t))
            ACF[0] = 10
            return ACF
        
def layer_generator(ACF, n, method):
    if method == "direct":
        T = sqrtm(toeplitz(ACF))
        def layer_gen(rng):
            a = np.matmul(T, rng.normal(size=n))
            b = np.matmul(T, rng.normal(size=n))
            return (a + 1j*b)/np.sqrt(2)
    elif method == "spectral":
        # === relationship between the dct and fft used here
        # rng = np.random.default_rng(42)
        # N = 5
        # x = rng.normal(size=N)
        # y = ifft(np.append(x, np.flip(x[1:-1])))
        # z = dct(x, type=1)/(2*(N - 1))
        # now we should have np.abs(y[:N] - z) == 0
        S = dct(ACF, type=1)
        g = np.sqrt(S/(2*(n - 1)))  # apply convolution theorem ans solve for the fourier transform of T
        full_g = np.append(g, np.flip(g[1:-1]))
        def layer_gen(rng):
            a = fft(full_g*rng.normal(size=2*(n - 1)))[:n]
            b = fft(full_g*rng.normal(size=2*(n - 1)))[:n]
            return (a + 1j*b)/np.sqrt(2)
    elif method == "hybrid":
        S = dct(ACF, type=1)
        T = dct(np.sqrt(S)/(2*(len(ACF) - 1)), type=1)  # apply convolution theorem and solve for T
        full_T = np.append(np.flip(T[1:]), T)
        def layer_gen(rng):
            a = rng.normal(size=n + 2*(len(ACF) - 1))
            b = rng.normal(size=n + 2*(len(ACF) - 1))
            c = (a + 1j*b)/np.sqrt(2)
            return np.correlate(full_T, c)
    return layer_gen

def calculate_ACFs(rng, layer_gen, samples, M, n, fractional_lag_steps):
    r = np.empty((samples, M))
    for i in range(samples):
        d = layer_gen(rng)[::fractional_lag_steps]
        for Delta in range(1, M + 1):
            r[i, Delta - 1] = np.real(np.vdot(d[Delta:], d[:-Delta]))/(len(d) - Delta)
    return r

if __name__ == "__main__":
    C = 20 #1370 
    n = 10*C
    h = 1
    M = 10
    N = 2
    fractional_lag_steps = 1 #137
    samples = 1000
    mode = "spread"
    method = "hybrid"

    t = np.arange(C)/fractional_lag_steps
    acf = gen_layer_ACF(t, h, N, mode)
    layer_gen = layer_generator(acf, n, method)
    rng = np.random.default_rng(50)
    res = calculate_ACFs(rng, layer_gen, samples, M, n, fractional_lag_steps)
    
    #%%
    plt.plot(t, acf)
    plt.errorbar(np.arange(1, M + 1),
                 np.mean(res, axis=0),
                 np.std(res, axis=0)/np.sqrt(samples),
                 capsize=5, marker="o", ls="None")
    norm_res = (np.mean(res, axis=0) - gen_layer_ACF(np.arange(1, M + 1), h, N, mode))*np.sqrt(samples)/np.std(res, axis=0)
    print(f"normalized residuals: {norm_res}")
    print(f"normalized chi2: {np.mean(norm_res**2):.1g}")
    plt.show()
    
    #%%
    r_table = np.zeros((M, M))
    for i in range(M - 1):
        for j in range(i + 1, M):
            r, p = scipy.stats.pearsonr(res[:, i], res[:, j])
            r_table[i, j] = r
            r_table[j, i] = r
    for i in range(M):
        r_table[i, i] = np.nan
    plt.imshow(r_table, extent=(0.5, M + 0.5, 0.5, M + 0.5))
    plt.xlabel("lag index")
    plt.ylabel("lag index")
    plt.colorbar()
    plt.clim(0, 1)
    plt.show()