# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import analyze_barker_modulation.decoder as decoder
import simulation_reader as sim
import signal_calc as sig

sim.read("sim_res/sim_1651933380.0185013")

off = len(sim.c)  # offset in number of pulses
w = (sim.n_code_tx - 2)*len(sim.c)  # number of pulses to process

print("decoding")
res = decoder.decode(sim.m, off, w, sim.c, sim.N, sim.M)


#%%

X, K, s = decoder.calc_approx_deconv_matrices(sim.g, len(sim.g), sim.rx_per_pulse)
res_deconv = np.zeros((sim.N, sim.M, sim.rx_per_pulse + len(sim.g)), dtype=np.complex128)
for h in range(sim.N):
    for Delta in range(sim.M):
        res_deconv[h, Delta] = X @ K.T @ res[h, Delta]

def plot_result(res, res_deconv, fname):
    x = np.arange(1, sim.M + 1)
    t_plot = np.linspace(0, sim.M)
    for h in range(sim.N):
        for w in [10]:
            layer_cnt = sim.N*(len(sim.g) + sim.rx_per_pulse - 1)
            plt.plot(t_plot, sig.gen_layer_ACF(t_plot, w + h*(len(sim.g) + sim.rx_per_pulse - 1) + (len(sim.g) - 1), layer_cnt, sim.mode), color="black", lw=1)
            plt.scatter(x, np.real(res[h, :, w])/len(sim.g), label=f"h={h}, w={w}")
            #plt.scatter(x, np.real(res_deconv[h, :, w]), label=f"h={h}, w={w}, deconv", marker="x")
    plt.legend()
    plt.show()

plot_result(res, res_deconv, "direct.pdf")

