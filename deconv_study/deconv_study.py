#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  8 17:48:57 2022

@author: frank
"""

import numpy as np
import matplotlib.pyplot as plt


assume_real_acf = True

def add_correlation(res_G_LL_UU, res_G_LU, res_G_UL, s1, s2, w1, w2, v, off_height, off_lag):
    delta = w1 - w2
    lag_index = off_lag + delta
    assert lag_index >= 0
    if s1 == s2:
        assert off_height + s1 >= 0
        res_G_LL_UU[off_height + s1, lag_index] += v
    if s1 == s2 + 1:
        assert off_height + s1 >= 0
        res_G_LU[off_height + s1, lag_index] += v
    if s1 + 1 == s2:
        assert off_height + s1 + 1 >= 0
        res_G_UL[off_height + s1 + 1, lag_index] += v

def calc_direct_deconv_point_spread(g, z):
    w = 0
    S = len(g)
    shape = (2*S - 1, 2*S - 1)
    res_G_LL_UU = np.zeros(shape)  # this contains the no-overlap range gates
    res_G_UL = np.zeros(shape)  # UL - overlapping range gates
    res_G_LU = np.zeros(shape)  # LU - overlapping range gates
    off_lag = S - 1
    off_height = 0
    for s1 in range(S):
        for s2 in range(S):
            for s3 in range(S):
                for s4 in range(S):
                    v = g[s1]*z[s2]*g[s3]*z[s4]
                    add_correlation(res_G_LL_UU, res_G_LU, res_G_UL,
                                    S - 1 - s1 + s2 + w, S - 1 - s3 + s4 + w,
                                    w + s2, w + s4,
                                    v, off_height, off_lag)
    return (res_G_LL_UU, res_G_UL + res_G_LU), off_lag, off_height


def calc_after_decode_point_spread(g):
    w = 0
    S = len(g)
    shape= (2*S - 1, 2*S - 1)
    res_G_LL_UU = np.zeros(shape)  # this contains the no-overlap range gates
    res_G_UL = np.zeros(shape)  # UL - overlapping range gates
    res_G_LU = np.zeros(shape)  # LU - overlapping range gates
    off_lag = S - 1
    off_height = 0
    for s1 in range(S):
        for s2 in range(S):
            v = g[s1]*g[s2]
            add_correlation(res_G_LL_UU, res_G_LU, res_G_UL,
                            S - 1 - s1 + w, S - 1 - s2 + w,
                            w, w,
                            v, off_height, off_lag)
    return (res_G_LL_UU, res_G_UL + res_G_LU), off_lag, off_height


def do_plot(r, off_lag, off_height):
    fig, axs = plt.subplots(2, 2 if assume_real_acf else 3)
    for ax_img, axs_plot, t in zip(axs[0], axs[1], r):
        img = ax_img.imshow(t, interpolation="None", origin="lower",
                            extent=(-off_lag - 0.5, -off_lag + t.shape[0] - 0.5,
                                    -off_height - 0.5, -off_height + t.shape[1] - 0.5))
        plt.colorbar(img, ax=ax_img)
        ax_img.set_xlabel("$\delta$ (fractional lag index)")
        ax_img.set_ylabel("s (sub height index)")
        fractional_lag_indices = np.arange(-off_lag , -off_lag + t.shape[1])
        for order in range(3):
            func = fractional_lag_indices**order
            contribution_by_height_indices = np.sum(t*func, axis=1)
            if not np.all(contribution_by_height_indices == 0):
                axs_plot.scatter(np.arange(t.shape[0]), contribution_by_height_indices, marker="x", label=f"order={order}")
        axs_plot.set_xlabel("s (sub height index)")
        axs_plot.set_ylabel("sum value ove $\delta$")
        axs_plot.legend()
    axs[0, 0].set_title("$G_\delta^{(L, L, s)} + G_\delta^{(U, U, s + 1)}$")
    if assume_real_acf:
        axs[0, 1].set_title("$G_\delta^{(U, L, s)} = G_\delta^{(L, U, s)}$")        
    else:
        axs[0, 1].set_title("$G_\delta^{(U, L, s)}$")
        axs[0, 2].set_title("$G_\delta^{(L, U, s)}$")
    plt.savefig("point_spread_funcs.pdf")
    plt.show()
    return r



if __name__ == "__main__":
    
    g_barker_13 = [+1, +1, +1, +1, +1, -1, -1, +1, +1, -1, +1, -1, +1]
    mod_17 = [+1, +1, +1, -1, -1, -1, -1, +1, +1, +1, -1, +1, +1, +1, -1, +1, +1, +1, -1, +1, +1, -1, +1, -1, -1, +1, -1]
    
    z = np.array([+1, +1, +1, +1, +1, -1, -1, +1, +1, -1, +1, -1, +1])
    o = calc_direct_deconv_point_spread(g_barker_13, z)
    do_plot(*o)
    
    #%%
    
    def find_best_inverse(z, g):
        (res_G_LL_UU, _), off_lag, off_height = calc_direct_deconv_point_spread(g, z)
        chi2 = (np.sum(res_G_LL_UU[len(g)-1]) - len(g)**2)**2 + np.max(np.max(res_G_LL_UU**4, axis=1)[:len(g)-1])
        print(chi2, z)
        return chi2
    
    from scipy.optimize import minimize
    
    res = minimize(find_best_inverse, g_barker_13, (g_barker_13))
    
    #%%
    o = calc_direct_deconv_point_spread(g_barker_13, res.x)
    do_plot(*o)
    
    
    #%%
    
    o = calc_after_decode_point_spread([-1, -1]*5)
    do_plot(*o)

