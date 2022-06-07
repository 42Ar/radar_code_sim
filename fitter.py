# -*- coding: utf-8 -*-

from deconv_study.psf_calculator import calc_PSF_array
import numpy as np
from calc_expect import calc_r, setup, index_to_alt, default_param_provider, calc_lag_vals
from theo import theo_cov_matrix_with_cutoff
from exp_io import load
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from scipy.linalg import inv
from scipy.interpolate import interp1d
from scipy.optimize import minimize

import sys
sys.path.insert(0, "../../special_curriculum/isr_sim")
import consts
from ionosphere import balance_charge


# note: negative ions are not in the model, thus we should stay above 80 km, see:
# "The negative-ion composition of the daytime D-region", L. THOMAS, P. M. GONDHALEKAR and M. R. BOWMAN
exp = "2022_5_6_11_0_51_1"
h_fit = 0
w_fit_start = 56
w_fit_stop = 67
acf_cutoff = 80
M_fit = 10
year_shift = 2

param_grid_start = None
param_grid_end = None
param_grid = {"Ne": 6, "Nn": 6}
dry = False
neglect_overlapping = False
neglect_correlation = True
plot_acf = False


def initial_param_vec():
    initial_vals, bounds = [], []
    for param, count in param_grid.items():
        x = np.linspace(param_grid_start, param_grid_end, count)
        for A in x:
            def_params = default_param_provider(A)
            if param == "Nn":
                def_val = float(def_params["N_N2"] + def_params["N_O2"] +
                                def_params["N_Ar"] + def_params["N_O"])
                bound = (0.1, 5)
            else:
                def_val = float(def_params[param])
                bound = (1e-10, None)
            initial_vals.append(def_val)
            bounds.append(bound)
    return initial_vals, bounds


def grid_param_provider(cur_param_vec):
    def provider(A):
        params = default_param_provider(A)
        cur_offset = 0
        for param, count in param_grid.items():
            x = np.linspace(param_grid_start, param_grid_end, count)
            y = cur_param_vec[cur_offset:cur_offset + count]
            cur_val = interp1d(x, y, copy=False, assume_sorted=True,
                               kind="nearest-up", fill_value="extrapolate")(A)
            # nearest-up
            if param == "Nn":
                def_val = (params["N_N2"] + params["N_O2"] + params["N_Ar"] +
                           params["N_O"])
                params["N_N2"] *= cur_val/def_val
                params["N_O2"] *= cur_val/def_val
                params["N_Ar"] *= cur_val/def_val
                params["N_O"] *= cur_val/def_val
            else:
                params[param] = cur_val
            cur_offset += count
        balance_charge(params)
        return params
    return provider


def calc_Sigma(acfs, w_fit_start, w_fit_stop, exp_info, acf_cutoff, M_fit):
    g = np.array(exp_info["g"])
    c = exp_info["c"]
    Pn0 = exp_info["B"]*exp_info["T_sys"]*consts.k  # exp_info["B"] is the *total* bandwidth
    cov_mat_size = (w_fit_stop - w_fit_start + 1)*M_fit
    Sigma = np.zeros((cov_mat_size, cov_mat_size))
    for w1 in range(w_fit_start, w_fit_stop + 1):
        for delta_w in range(min(2*len(g), w_fit_stop - w1 + 1)):
            acfs_for_sub_mat = acfs[:, :, w1 - w_fit_start, delta_w]
            sub_mat = theo_cov_matrix_with_cutoff(h_fit, c, exp_info["pulses"],
                                                  acf_cutoff, 1,
                                                  acfs_for_sub_mat, M_fit)
            off = (w1 - w_fit_start)*M_fit
            a = slice(off, off + M_fit)
            b = slice(off + delta_w*M_fit, off + delta_w*M_fit + M_fit)
            if delta_w < len(g):
                Pn = Pn0*np.sum(g[delta_w:]*g[:len(g) - delta_w])
                sub_mat += np.eye(M_fit)*(Pn**2/(2*exp_info["pulses"]))
            Sigma[a, b] = sub_mat
            Sigma[b, a] = sub_mat.T
    #TODO add sky noise
    return Sigma


def calc_neglogL(p, p0, x, exp_config, w_fit_start, w_fit_stop, PSF_array,
                 acf_cutoff, acf_len, h_fit, M_fit, neglect_correlation,
                 debugplot_result=False):
    params = p*p0
    if neglect_correlation:
        acfs = np.array([calc_r(h_fit, np.arange(M_fit + 1), w_fit_start,
                                w_fit_stop, exp_info, PSF_array,
                                grid_param_provider(params),
                                neglect_overlapping=neglect_overlapping,
                                no_delta_w=True)])
        mu = acfs.squeeze()[1:M_fit + 1].T.ravel()
        var = acfs.squeeze()[0].repeat(M_fit)**2
        neglogL = np.sum((mu - x)**2/var)
    else:
        acfs = np.array([calc_r(h_fit, np.arange(acf_len), w_fit_start,
                                w_fit_stop, exp_info, PSF_array,
                                grid_param_provider(params),
                                neglect_overlapping=neglect_overlapping)])
        Sigma = calc_Sigma(acfs, w_fit_start, w_fit_stop, exp_info, acf_cutoff,
                           M_fit)
        mu = acfs[h_fit, 1:M_fit + 1, :, 0].T.ravel()
        d = mu - x
        prec = inv(Sigma)
        neglogL = d@prec@d
    print(f"-log(L)={neglogL:1.2e}:" + ",".join(f"{v:5.2g}" for v in p))
    if debugplot_result:
        fig, axs = plt.subplots(2, 1, sharex=True)
        plt.subplots_adjust(hspace=0)
        x_plot = np.arange(len(mu))
        std = np.sqrt(var if neglect_correlation else np.diag(Sigma))
        for xp, yp, ye in zip(x_plot.reshape(-1, M_fit), mu.reshape(-1, M_fit), std.reshape(-1, M_fit)):
            axs[0].errorbar(xp, yp, ye, capsize=5, c="black", lw=1, ls="--")
        axs[0].scatter(x_plot, x)
        axs[0].set_ylabel("decoded rx power in W")
        axs[1].scatter(x_plot, (x - mu)/std)
        axs[1].set_xlabel("datapoint index")
        axs[1].set_ylabel("normalized residuals")
        axs[1].axhspan(-1, 1, color="grey", alpha=0.5)
        fig.show()
        
        if not neglect_correlation:
            fig, ax = plt.subplots()
            ax.matshow(Sigma)
            ax.set_xlabel("datapoint index")
            ax.set_ylabel("datapoint index")
            fig.show()
        
        if plot_acf:
            fig, ax = plt.subplots()
            ax.plot(calc_lag_vals(exp_config, np.arange(M_fit))[:, 0],
                    acfs[h_fit, :M_fit, (w_fit_stop - w_fit_start)//2, 0],
                    marker="o", ls="None")
            fig.show()
    return neglogL

exp_data, exp_info = load(exp)
assert M_fit <= exp_info["M"]
assert h_fit == 0, "currently, only h_fit = 0 supported"
start_time = datetime(exp_info["year"], exp_info["month"], exp_info["day"],
                      exp_info["hour"], exp_info["minute"], exp_info["second"])
mean_time = start_time + timedelta(-365*year_shift,
                                   (exp_info["start"] + exp_info["end"])/2)
setup(mean_time, exp_info["P_mean"], 60e3, 150e3, 0.3e3)
PSF_array = calc_PSF_array(exp_info["g"])
c = exp_info["c"]
acf_len = len(c)*((acf_cutoff + exp_info["M"] + 2*len(c))//len(c)) + exp_info["M"]
start_h = index_to_alt(exp_info, h_fit, w_fit_start + len(exp_info["g"]) - 1)
end_h = index_to_alt(exp_info, h_fit, w_fit_stop + len(exp_info["g"]) - 1)
print(f"fitting {start_h*1e-3:.0f} km to {end_h*1e-3:.0f} km")
x = np.real(exp_data[h_fit, :M_fit, w_fit_start:w_fit_stop + 1].T.ravel())
if param_grid_start is None:
    param_grid_start = start_h
if param_grid_end is None:
    param_grid_end = end_h
p0, bounds = initial_param_vec()
p = np.ones(len(p0))
if dry:
    calc_neglogL(p, p0, x, exp_info, w_fit_start, w_fit_stop, PSF_array,
                 acf_cutoff, acf_len, h_fit, M_fit, neglect_correlation,
                 debugplot_result=True)
else:
    args = (p0, x, exp_info, w_fit_start, w_fit_stop, PSF_array, acf_cutoff,
            acf_len, h_fit, M_fit, neglect_correlation)
    res = minimize(calc_neglogL, p, args, bounds=bounds)

#%%
calc_neglogL(res.x, p0, x, exp_info, w_fit_start, w_fit_stop, PSF_array,
             acf_cutoff, acf_len, h_fit, M_fit, neglect_correlation=False,
             debugplot_result=True)



