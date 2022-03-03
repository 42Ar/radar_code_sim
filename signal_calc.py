# -*- coding: utf-8 -*-

from config import N, n, K, method
import numpy as np
from scipy.linalg import sqrtm, toeplitz
from scipy.fftpack import dct, fft
import matplotlib.pyplot as plt


mode = "spread"

def gen_layer_ACF(t, h):
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
        
def layer_generator(ACF):
    if method == "direct":
        T = sqrtm(toeplitz(ACF))
        def layer_gen(rand):
            a = np.matmul(T, rand.normal(size=n))
            b = np.matmul(T, rand.normal(size=n))
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
        def layer_gen(rand):
            a = fft(full_g*rand.normal(size=2*(n - 1)))[:n]
            b = fft(full_g*rand.normal(size=2*(n - 1)))[:n]
            return (a + 1j*b)/np.sqrt(2)
    elif method == "hybrid":
        ACF = ACF[:K]
        S = dct(ACF, type=1)
        T = dct(np.sqrt(S)/(2*(K - 1)), type=1)  # apply convolution theorem ans solve for T
        full_T = np.append(np.flip(T[1:]), T)
        def layer_gen(rand):
            a = rand.normal(size=n + 2*(K - 1))
            b = rand.normal(size=n + 2*(K - 1))
            c = (a + 1j*b)/np.sqrt(2)
            return np.correlate(full_T, c)
    return layer_gen

def calculate_ACFs(rng):
    r = np.empty((samples, M))
    for i in range(samples):
        d = layer_gen(rng)
        for Delta in range(1, M + 1):
            r[i, Delta - 1] = np.real(np.vdot(d[Delta:], d[:-Delta]))/(n - Delta)
    return r

if __name__ == "__main__":
    t = np.arange(n)
    h = 0
    acf = gen_layer_ACF(t, h)
    layer_gen = layer_generator(acf)
    M = 5
    samples = 10000
    rng = np.random.default_rng(45)
    r = calculate_ACFs(rng)
    t_plot = np.linspace(0, M, 100)
    plt.plot(t_plot, gen_layer_ACF(t_plot, h), color="black", lw=1)
    plt.errorbar(np.arange(1, M + 1),
                 np.mean(r, axis=0),
                 np.std(r, axis=0)/np.sqrt(samples),
                 capsize=5, marker="o", ls="None")
    print("normalized residuals:", (np.mean(r, axis=0) - acf[1:M+1])*np.sqrt(samples)/np.std(r, axis=0))
    plt.show()


