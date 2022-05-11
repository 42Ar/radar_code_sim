# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import signal_calc as sig
import scipy.linalg as linalg
import simulation_reader as sim
import analyze_barker_modulation.decoder as decoder

sim_small = "sim_res/sim_1651934695.3934464"
sim_rx_samples_as_measured = "sim_res/sim_1651968446.2103086"
sim_rx_samples_as_measured_no_correlation = "sim_res/sim_1652054457.6305187"
sim.read(sim_rx_samples_as_measured_no_correlation)

sim.m = sim.m[56000//4:, :]
sim.n_code_tx //= 2

print(f"simulation method: {sim.gen_method}")
off = len(sim.c)  # offset in number of pulses
w = (sim.n_code_tx - 2)*len(sim.c)  # number of pulses to process
layer_cnt = sim.N*(len(sim.g) + sim.rx_per_pulse - 1)


#%%

print("decoding")
d = np.array([np.correlate(rxdata, sim.g, mode="valid") for rxdata in sim.m])
res = decoder.decode(d, off, w, sim.c, sim.N, sim.M)

#%%

K = sim.rx_per_pulse - len(sim.g) + 1  # basically the number of valid sub height indices

def plot_result(res, fname):
    x = np.arange(1, sim.M + 1)
    fractional_lag_step = 1/(len(sim.g) + sim.rx_per_pulse - 1)
    K = 10*(len(sim.g) + sim.rx_per_pulse - 1)
    t_sim = np.arange(K)*fractional_lag_step
    for h in range(sim.N):
        for w in [0]:
            acf = sig.gen_layer_ACF(t_sim, w + h*(len(sim.g) + sim.rx_per_pulse - 1) + len(sim.g) - 1, layer_cnt, sim.mode)
            plt.plot(t_sim, acf, color="black")
            plt.scatter(x, np.real(res[h, :, w]), label=f"h={h}, w={w}")
    plt.legend()
    plt.savefig(fname)
    plt.show()

plot_result(res/len(sim.g)**2, "direct.pdf")

#%%

A = np.correlate(sim.g, sim.g, mode="full")**2
conv_mat = np.zeros((K, K), dtype=np.int32)
for i in range(K):
    l = min(len(sim.g) - 1, i)
    r = min(len(sim.g), K - i)
    conv_mat[i, i - l:i+r] = A[len(sim.g) - 1 - l:len(sim.g) - 1 + r]
A = linalg.inv(conv_mat)
res_deconv = np.zeros(res.shape, res.dtype)
for h in range(sim.N):
    for m in range(sim.M):
        res_deconv[K*h:K*(h + 1), m] = np.matmul(A, res[K*h:K*(h + 1), m])
plot_result(res_deconv, "deconvolved.pdf")
