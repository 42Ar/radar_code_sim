# -*- coding: utf-8 -*-

import sys
sys.path.insert(0, "../../special_curriculum/isr_sim")

import ionosphere
import config
import ACF
import consts
from scipy.interpolate import interp1d
import numpy as np
import signal_calc
from deconv_study.psf_calculator import get_PSF, calc_delta_min, calc_delta_max


def setup(time, power, min_h, max_h, step_h):
    config.time = time
    config.P = power
    ionosphere.precalc_ionosphere(min_h, max_h, step_h)


def index_to_alt(exp_config, h, H):
    return consts.c*(h*exp_config["tau_IPP"] +
                     exp_config["tau_d"] +
                     (H + 0.5)*exp_config["tau"])/2


def default_param_provider(A):
    return ionosphere.calc_spec_params(A)


def calc_lag_vals(exp_config, Delta_vals):
    S = len(exp_config["g"])
    delta = np.arange(-S + 1, S)
    return (exp_config["tau_IPP"]*Delta_vals[:, np.newaxis] +
            exp_config["tau"]*delta)


def calc_expect_on_alt(h, H, Delta_vals, exp_config, param_provider):
    S = len(exp_config["g"])
    if "type" in exp_config and exp_config["type"] == "test_sim":
        layer_cnt = exp_config["layer_cnt"]
        mode = exp_config["mode"]
        W = exp_config["W"]
        steps_per_lag = S + W - 1
        fractional_lag_step = 1/(S + W - 1)
        delta = np.arange(-S + 1, S)
        G_LL = [signal_calc.gen_layer_ACF(Delta + delta*fractional_lag_step,
                                          h*steps_per_lag + H, layer_cnt,
                                          mode)/2 for Delta in Delta_vals]
        G_LU = np.zeros((len(Delta_vals), 2*S - 1))
    else:
        A = index_to_alt(exp_config, h, H)
        a = consts.c*exp_config["tau"]/2
        spec_params = param_provider(A)
        dt, R = ACF.calc_R(A, spec_params, a/3)
        acf = interp1d(dt, R, assume_sorted=True, copy=False, fill_value=0,
                       bounds_error=False)
        lag_vals = calc_lag_vals(exp_config, Delta_vals)
        G_LL = acf(np.abs(lag_vals))
        G_LU = acf(np.abs(lag_vals + exp_config["tau"]/2))
    return [G_LL, G_LU]


def calc_r(h, Delta_vals, w_start, w_stop, exp_config, psf_array,
           param_provider, neglect_overlapping=False, debugplot=False,
           no_delta_w=False):
    S = len(exp_config["g"])
    W = exp_config["W"]
    assert w_stop >= w_start and w_start >= 0 and w_stop <= W - S
    abs_q_max = 2*(S - 1)
    Delta_vals = np.array(Delta_vals)
    all_alts = np.array([calc_expect_on_alt(h, H, Delta_vals, exp_config,
                                            param_provider)
                         for H in range(w_start, w_stop + 2*S)])
    res = []
    for Delta in Delta_vals:
        alts = all_alts[:, :, Delta]
        res_Delta = []
        for w1 in range(w_stop - w_start + 1):
            res_w = []
            for delta_w in range(1 if no_delta_w else 2*S):
                r = 0
                q = -delta_w
                if abs(q) <= abs_q_max:
                    PSF = get_PSF(psf_array, q)
                    dmin = calc_delta_min(q, S)
                    dmax = calc_delta_max(q, S)
                    if debugplot and Delta == 1 and delta_w == 0:
                        import matplotlib.pyplot as plt
                        fig, axs = plt.subplots(1, 2)
                        axs[0].imshow(PSF.T, origin="lower")
                        axs[1].imshow(alts[w1:w1 + 2*S - 1, 0, S - 1 + dmin:S + dmax], origin="lower")
                        fig.show()
                    r += np.sum((alts[w1 + 1:w1 + 2*S, 0, S - 1 + dmin:S + dmax] +
                                 alts[w1:w1 + 2*S - 1, 0, S - 1 + dmin:S + dmax])*PSF.T)
                if not neglect_overlapping:
                    q = -delta_w + 1
                    if abs(q) <= abs_q_max:
                        PSF = get_PSF(psf_array, q)
                        dmin = calc_delta_min(q, S)
                        dmax = calc_delta_max(q, S)
                        r += np.sum(alts[w1:w1 + 2*S - 1, 1, S - 1 + dmin:S + dmax]*PSF.T)
                    q = -delta_w - 1
                    if abs(q) <= abs_q_max:
                        PSF = get_PSF(psf_array, q)
                        dmin = calc_delta_min(q, S)
                        dmax = calc_delta_max(q, S)
                        r += np.sum(alts[w1 + 1:w1 + 2*S, 1, S - 1 + dmin:S + dmax]*PSF.T)
                res_w.append(r)
            res_Delta.append(res_w)
        res.append(res_Delta)
    return np.array(res)