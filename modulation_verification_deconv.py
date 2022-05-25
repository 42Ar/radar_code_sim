# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import analyze_barker_modulation.decoder as decoder
import simulation_reader as sim
import signal_calc as sig

#sim_small = "sim_res/sim_1651934695.3934464"
sim_rx_samples_as_measured = "sim_res/sim_1651968446.2103086"
sim.read(sim_rx_samples_as_measured)

off = len(sim.c)  # offset in number of pulses
w = (sim.n_code_tx - 2)*len(sim.c)  # number of pulses to process

print("decoding")
d = decoder.decode(sim.m, off, w, sim.c, sim.N, sim.M)


#%%

X, K, s = decoder.calc_approx_deconv_matrices(sim.g, sim.rx_per_pulse, 50)
#plt.imshow(X); plt.colorbar()
deconv = decoder.deconvolute_matrix_method_approx(d, X, K)


def plot_result(d, deconv, fname):
    x = np.arange(1, sim.M + 1)
    t_plot = np.linspace(0, sim.M)
    for h in range(sim.N):
        for w in [sim.rx_per_pulse//2]: #0, sim.rx_per_pulse - 1, 
            layer_cnt = sim.N*(len(sim.g) + sim.rx_per_pulse - 1)
            plt.plot(t_plot, sig.gen_layer_ACF(t_plot, w + h*(len(sim.g) + sim.rx_per_pulse - 1) + (len(sim.g) - 1)/2, layer_cnt, sim.mode), color="black", lw=1)
            plt.scatter(x, np.real(d[h, :, w])/len(sim.g), label=f"h={h}, w={w}")
            plt.scatter(x, np.real(deconv[h, :, w]), marker="x")
    plt.legend()
    plt.show()

plot_result(d, deconv, "direct.pdf")

