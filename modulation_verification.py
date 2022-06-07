# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import signal_calc as sig
import simulation_reader as sim
import analyze_barker_modulation.decoder as decoder
from theo import theo_cov_matrix_with_cutoff
from calc_expect import calc_r
from deconv_study.psf_calculator import calc_PSF_array


sim_small = "sim_res/sim_1651934695.3934464"
sim_rx_samples_as_measured = "sim_res/sim_1651968446.2103086"
sim_rx_samples_as_measured_no_correlation = "sim_res/sim_1652054457.6305187"
sim_rx_samples_as_measured_no_correlation_long_cutoff = "sim_res/sim_1652394386.0852432"
sim.read(sim_rx_samples_as_measured_no_correlation_long_cutoff)
assert sim.m.shape[1] == sim.rx_per_pulse

print(f"simulation method: {sim.gen_method}")
steps_per_lag = len(sim.g) + sim.rx_per_pulse - 1
fractional_lag_step = 1/(len(sim.g) + sim.rx_per_pulse - 1)
layer_cnt = sim.N*steps_per_lag

#%%

rx_per_sample = len(sim.c)*5
rx_per_sample_total = len(sim.c)*6  # chosen so that the next sample is at even multiple of the code (for convienience)
n_samples = sim.m.shape[0]//rx_per_sample_total

print("decoding")
samples = []
for s in range(n_samples):
    off = s*rx_per_sample_total
    sample_data = sim.m[off:off + rx_per_sample_total]
    m = decoder.filter_sub_pulse(sample_data, sim.g)
    samples.append(decoder.decode(m, 0, rx_per_sample, sim.c, sim.N, sim.M))
samples = np.array(samples)


#%%

PSF_array = calc_PSF_array(sim.g)
exp_config = {
    "g": sim.g,
    "W": sim.rx_per_pulse,
    "type": "test_sim",
    "layer_cnt": layer_cnt,
    "mode": sim.mode,
}

K_whole_lags = sim.cutoff//(len(sim.g) + sim.rx_per_pulse - 1)
n_whole_lags = 2*len(sim.c) + sim.M

acfs = np.array([calc_r(h, np.arange(n_whole_lags), 0,
                        sim.rx_per_pulse - len(sim.g),
                        exp_config, PSF_array, None) for h in range(sim.N)])


#%%

def plot_mean():
    x = np.arange(1, sim.M + 1)
    for w in [60]:
        for h in [0]:
            cov = theo_cov_matrix_with_cutoff(h, sim.c, rx_per_sample,
                                              K_whole_lags, sim.N,
                                              acfs[:, :, w, 0], sim.M)       
            tsim = np.arange(sim.cutoff)*fractional_lag_step
            acf = sig.gen_layer_ACF(tsim, h*steps_per_lag + w + len(sim.g) - 1,
                                    layer_cnt, sim.mode)
            plt.plot(tsim, acf*len(sim.g)**2, color="black")
            
            expected = acfs[h, 1:sim.M + 1, w, 0]
            measured = np.mean(np.real(samples[:, h, :, w]), axis=0)
            plt.errorbar(x, expected, np.sqrt(np.diag(cov)/n_samples),
                         ls="None", capsize=5, marker="x", c="black")
            plt.scatter(x, measured, marker="o", label=f"h={h}, w={w}")
    plt.legend()
    plt.title("resulting mean plot")
    plt.xlim(0, sim.M + 0.5)
    plt.show()
    
plot_mean()

#%%

def plot_std():
    # the w = 0, h = 1 point is probably off just because there is some
    # correlation between h = 0 and h = 1 in the simulation which is not
    # taken into account in the calculation
    # it does not occure in the measurement
    x = np.arange(1, sim.M + 1)
    for w in [0, 10, 50, 80, 100]:
        for h in [0, 1]:
            cov = theo_cov_matrix_with_cutoff(h, sim.c, rx_per_sample,
                                              K_whole_lags, sim.N,
                                              acfs[:, :, w, 0], sim.M)
            expected = np.sqrt(np.diag(cov))
            measured = np.std(np.real(samples[:, h, :, w]), axis=0)
            plt.scatter(x, expected/measured)
    plt.axvline(1)
    plt.grid()

plot_std()

#%%

def plot_corr_ratio():
    h = 0
    w = 80
    cov_theo = theo_cov_matrix_with_cutoff(h, sim.c, rx_per_sample,
                                           K_whole_lags, sim.N,
                                           acfs[:, :, w, 0], sim.M)    
    cov_obs = np.cov(np.real(samples[:, h, :, w]), rowvar=False)
    plt.matshow(np.corrcoef(cov_obs) - np.corrcoef(cov_theo))
    plt.colorbar()
    plt.title("correlation matrix sim/theo")
    plt.xlabel(r"$\Delta_1$")
    plt.ylabel(r"$\Delta_2$")
    plt.show()


plot_corr_ratio()

#%%

def plot_cov_with_nonzero_delta_w():
    h = 0
    w = 80
    cov_theo = theo_cov_matrix_with_cutoff(h, sim.c, rx_per_sample,
                                           K_whole_lags, sim.N,
                                           acfs[:, :, w, 2], sim.M)    
    #cov_obs = np.cov(np.real(samples[:, h, :, w]), rowvar=False)
    plt.matshow(cov_theo)
    plt.colorbar()
    plt.xlabel(r"$\Delta_1$")
    plt.ylabel(r"$\Delta_2$")
    plt.show()


plot_cov_with_nonzero_delta_w()


