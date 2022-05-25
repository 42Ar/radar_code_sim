#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  8 17:48:57 2022

@author: frank
"""

import numpy as np
import matplotlib.pyplot as plt
from deconv_study.psf_calculator import calc_PSF
from scipy.optimize import minimize

assume_real_acf = False

def do_plot(r, off_lag):
    fig, axs = plt.subplots(2, 3)
    for ax_img, axs_plot, t in zip(axs[0], axs[1], r):
        img = ax_img.imshow(t, interpolation="None", origin="lower",
                            extent=(-off_lag - 0.5, -off_lag + t.shape[0] - 0.5,
                                    -0.5, t.shape[1] - 0.5))
        plt.colorbar(img, ax=ax_img)
        ax_img.set_xlabel(r"$\delta$ (fractional lag index)")
        ax_img.set_ylabel(r"H (sub height index)")
        fractional_lag_indices = np.arange(-off_lag , -off_lag + t.shape[1])
        for order in range(3):
            func = fractional_lag_indices**order
            contribution_by_height_indices = np.sum(t*func, axis=1)
            if not np.all(contribution_by_height_indices == 0):
                axs_plot.scatter(np.arange(t.shape[0]), contribution_by_height_indices, marker="x", label=f"order={order}")
        axs_plot.set_xlabel("H (sub height index)")
        axs_plot.set_ylabel("sum over $\delta$ axis")
        axs_plot.legend()
    axs[0, 0].set_title(r"$G_\delta^{(L, L, H)} + G_\delta^{(U, U, H + 1)}$")
    axs[0, 1].set_title(r"$G_\delta^{(L, U, H)}$")
    axs[0, 2].set_title(r"$\left(G_\delta^{(L, U, H + 1)}\right)^*$")
    plt.savefig("deconv_study/point_spread_funcs.pdf")
    plt.show()
    return r


#%%

g_barker_13 = [+1, +1, +1, +1, +1, -1, -1, +1, +1, -1, +1, -1, +1]

r = [calc_PSF(q, g_barker_13) for q in [0, 1, -1]]
do_plot(r, len(g_barker_13) - 1)

