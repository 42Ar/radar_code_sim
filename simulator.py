# -*- coding: utf-8 -*-
import numpy as np
import signal_calc as sig
import time
import json


N = 5
M = 4
c = [-1, -1, -1, -1, +1, -1, +1, -1, +1, -1, -1, -1, +1, +1, -1, +1, -1, -1, +1, -1, -1, -1, +1, +1]
g = [+1, +1, +1, +1, +1, -1, -1, +1, +1, -1, +1, -1, +1]
rx_per_pulse = 2*len(g)  # number of rx samples per pulse
n_code_tx = 40  # number of full code transmissions (number of pulses in units of len(c))
gen_method = "correlated_symmetric"  # simulate correlation between neighbouring range gates
mode = "spread"


print("writing info file")
name = f"sim_res/sim_{time.time()}"
info = {"N": N, "M": M, "c": c, "g": g, "gen_method": gen_method,
        "mode": mode, "rx_per_pulse": rx_per_pulse, "n_code_tx": n_code_tx}
with open(f"{name}.json", 'w') as inf_file:
    json.dump(info, inf_file, indent=4)
    

print("generating")
n = (len(g) + rx_per_pulse - 1)*len(c)*n_code_tx  # total number of samples rx + tx
rng = np.random.default_rng(42)
t = np.arange(n)/(len(g) + rx_per_pulse - 1)
layer_cnt = N*(len(g) + rx_per_pulse - 1)
K_corrected = 10*(len(g) + rx_per_pulse - 1)  # define greater K (ACF cutoff) as time steps are shorter
if gen_method == "uncorrelated":  # overlapping ranges gates are uncorrelated
    layers = np.zeros((layer_cnt, n), dtype=np.complex128)
    for h in range(layer_cnt):
        print(f"{h + 1}/{layer_cnt}")
        acf = sig.gen_layer_ACF(t, h, layer_cnt, mode)
        layers[h] = sig.layer_generator(acf, n, K_corrected)(rng)
elif gen_method == "correlated_simple":  # average over two range gates to create correlation
    layers = np.zeros((layer_cnt + 1, n), dtype=np.complex128)
    for h in range(layer_cnt + 1):
        print(f"gen {h + 1}/{layer_cnt + 1}")
        acf = sig.gen_layer_ACF(t, h, layer_cnt + 1, mode)
        layers[h] = sig.layer_generator(acf, n, K_corrected)(rng)
    for h in range(layer_cnt):
        print(f"avg {h + 1}/{layer_cnt}")
        layers[h] = (layers[h] + layers[h + 1])/np.sqrt(2)
    layers = layers[:-1]
elif gen_method == "correlated_symmetric":  # generates one signal for each half range gate
    layers = np.zeros((layer_cnt*2, n*2), dtype=np.complex128)
    t = np.arange(2*n)/(len(g) + rx_per_pulse - 1)/2
    for h in range(layer_cnt*2):
        print(f"gen {h + 1}/{layer_cnt*2}")
        acf = sig.gen_layer_ACF(t, h, layer_cnt*2, mode)
        layers[h] = sig.layer_generator(acf, 2*n, K_corrected*2)(rng)
    for h in range(layer_cnt):
        print(f"avg {h + 1}/{layer_cnt}")
        layers[h, :n] = (layers[2*h, ::2] + layers[2*h + 1, 1::2])/np.sqrt(2)
    layers = layers[:layer_cnt, :n]


print("encoding")
columns = layers.reshape((layer_cnt, len(c)*n_code_tx, rx_per_pulse + len(g) - 1))[:, :, :rx_per_pulse]
columns = np.swapaxes(columns, 0, 1)
for ci, col in enumerate(columns):
    mod = np.outer(np.roll(np.flip(c), ci + 1)[:N], np.append(np.flip(g), np.zeros(rx_per_pulse - 1))).ravel()
    for cj in range(col.shape[1]):
        col[:, cj] *= np.roll(mod, cj)
m = columns.sum(axis=1)
np.save(name, m)

