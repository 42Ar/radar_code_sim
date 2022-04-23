# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import signal_calc as sig

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
sim_corr = False  # simulate correlation between neighbouring range gates

print("generating")
rng = np.random.default_rng(42)
t = np.arange(n)/(len(g) + rx_per_pulse - 1)
layer_cnt = N*(len(g) + rx_per_pulse - 1)
K_corrected = 10*(len(g) + rx_per_pulse - 1)  # need greater K (ACF cutoff) as time steps are shorter
if sim_corr:
    if layer_cnt%2 == 0:
        layer_cnt += 1
    layers = np.zeros((layer_cnt, n), dtype=np.complex128)
    for h in range(0, layer_cnt, 2):
        acf = sig.gen_layer_ACF(t, h, layer_cnt)
        layers[h] = sig.layer_generator(acf, n, K_corrected)(rng)
    for h in range(1, layer_cnt, 2):
        layers[h, :-1] = (layers[h+1, :-1] + layers[h-1, :-1] + layers[h+1, 1:] + layers[h-1, 1:])/4
else:
    layers = np.zeros((layer_cnt, n), dtype=np.complex128)
    for h in range(layer_cnt):
        acf = sig.gen_layer_ACF(t, h, layer_cnt)
        layers[h] = sig.layer_generator(acf, n, K_corrected)(rng)
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

valid_sub_i = rx_per_pulse - len(g) + 1
x = np.arange(1, M + 1)
t_plot = np.linspace(0, M)
for h in range(N):
    for w in [0, valid_sub_i - 1]:
        plt.plot(t_plot, sig.gen_layer_ACF(t_plot, w + h*(len(g) + rx_per_pulse - 1) + len(g) - 1, layer_cnt), color="black", lw=1)
        plt.scatter(x, np.real(res[h*valid_sub_i + w])/len(g)**2, label=f"h={h}, w={w}")
plt.legend()
plt.show()



