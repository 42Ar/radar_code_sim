# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import signal_calc as sig
import scipy.linalg as linalg

#M = 5
#N = 7
#c = [-1, +1, +1, -1, +1, -1, +1, +1, -1, -1, +1, +1, -1, -1, -1, +1, -1, +1, -1, +1, +1, -1, +1, -1, +1, +1, +1, +1, +1, -1, +1, +1, +1, -1, -1, -1, +1, +1, +1, +1]
N = 5
M = 4
c = [-1, -1, -1, -1, +1, -1, +1, -1, +1, -1, -1, -1, +1, +1, -1, +1, -1, -1, +1, -1, -1, -1, +1, +1]
g = [+1, +1, +1, +1, +1, -1, -1, +1, +1, -1, +1, -1, +1]
rx_per_pulse = 2*len(g)  # number of rx samples per pulse
n_code_tx = 2000  # number of full code transmissions (number of pulses in units of len(c))
n = (len(g) + rx_per_pulse - 1)*len(c)*n_code_tx  # total number of samples rx + tx
off = len(c)  # offset in number of pulses
w = (n_code_tx - 2)*len(c)  # number of pulses to process
gen_method = "correlated_symmetric"  # simulate correlation between neighbouring range gates

print("generating")
rng = np.random.default_rng(42)
t = np.arange(n)/(len(g) + rx_per_pulse - 1)
layer_cnt = N*(len(g) + rx_per_pulse - 1)
K_corrected = 10*(len(g) + rx_per_pulse - 1)  # define greater K (ACF cutoff) as time steps are shorter
if gen_method == "uncorrelated":  # overlapping ranges gates are uncorrelated
    layers = np.zeros((layer_cnt, n), dtype=np.complex128)
    for h in range(layer_cnt):
        acf = sig.gen_layer_ACF(t, h, layer_cnt)
        layers[h] = sig.layer_generator(acf, n, K_corrected)(rng)
elif gen_method == "correlated_simple":  # average over two range gates to create correlation
    layers = np.zeros((layer_cnt + 1, n), dtype=np.complex128)
    for h in range(layer_cnt + 1):
        acf = sig.gen_layer_ACF(t, h, layer_cnt + 1)
        layers[h] = sig.layer_generator(acf, n, K_corrected)(rng)
    for h in range(layer_cnt):
        layers[h] = (layers[h] + layers[h + 1])/np.sqrt(2)
    layers = layers[:-1]
elif gen_method == "correlated_symmetric":  # generates one signal for each half range gate
    layers = np.zeros((layer_cnt*2, n*2), dtype=np.complex128)
    t = np.arange(2*n)/(len(g) + rx_per_pulse - 1)/2
    for h in range(layer_cnt*2):
        acf = sig.gen_layer_ACF(t, h, layer_cnt*2)
        layers[h] = sig.layer_generator(acf, 2*n, K_corrected*2)(rng)
    for layers in range(layer_cnt):
        layers[h, :n] = (layers[2*h, ::2] + layers[2*h + 1, 1::2])/np.sqrt(2)
    layers = layers[:layer_cnt, :n]

print("encoding")
columns = layers.reshape((layer_cnt, len(c)*n_code_tx, rx_per_pulse + len(g) - 1))[:, :, :rx_per_pulse]
columns = np.swapaxes(columns, 0, 1)
for ci, col in enumerate(columns):
    mod = np.outer(np.roll(np.flip(c), ci + 1)[:N], np.append(np.flip(g), np.zeros(rx_per_pulse - 1))).ravel()
    for cj in range(col.shape[1]):
        col[:, cj] *= np.roll(mod, cj)

#%%

print("decoding")
m = np.sum(columns, axis=1)
d = np.array([np.correlate(rxdata, g, mode="valid") for rxdata in m])
res = []
cc = np.tile(c, n_code_tx)
for h in range(N):
    for k in range(d.shape[1]):
        d1 = d[off + h:off + h + w, k]*cc[off:off + w]
        rr = []
        for Delta in range(1, M + 1):
            d2 = d[off + h + Delta:off + h + Delta + w, k]*cc[off + Delta:off + Delta + w]
            rr.append(np.vdot(d2, d1)/w)
        res.append(rr)
res = np.array(res)

#%%

K = rx_per_pulse - len(g) + 1  # basically the number of valid sub height indices

def plot_result(res, fname):
    x = np.arange(1, M + 1)
    t_plot = np.linspace(0, M)
    for h in range(N):
        for w in [0, K - 1]:
            plt.plot(t_plot, sig.gen_layer_ACF(t_plot, w + h*(len(g) + rx_per_pulse - 1) + len(g) - 1, layer_cnt), color="black", lw=1)
            plt.scatter(x, np.real(res[h*K + w]), label=f"h={h}, w={w}")
    plt.legend()
    plt.savefig(fname)
    plt.show()

plot_result(res/len(g)**2, "direct.pdf")

#%%

A = np.correlate(g, g, mode="full")**2
conv_mat = np.zeros((K, K), dtype=np.int32)
for i in range(K):
    l = min(len(g) - 1, i)
    r = min(len(g), K - i)
    conv_mat[i, i - l:i+r] = A[len(g) - 1 - l:len(g) - 1 + r]
A = linalg.inv(conv_mat)
res_deconv = np.zeros(res.shape, res.dtype)
for h in range(N):
    for m in range(M):
        res_deconv[K*h:K*(h + 1), m] = np.matmul(A, res[K*h:K*(h + 1), m])
plot_result(res_deconv, "deconvolved.pdf")
