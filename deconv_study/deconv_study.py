#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  8 17:48:57 2022

@author: frank
"""

import numpy as np
import matplotlib.pyplot as plt
from deconv_study.psf_calculator import calc_PSF, calc_delta_min


def do_plot(r, q_vals, S):
    fig, axs = plt.subplots(2, len(q_vals))
    for ax_img, axs_plot, t, q in zip(axs[0], axs[1], r, q_vals):
        delta_min = calc_delta_min(q, S)
        img = ax_img.imshow(t.T, interpolation="None", origin="lower",
                            extent=(delta_min - 0.5, delta_min + t.shape[0] - 0.5,
                                    -0.5, t.shape[1] - 0.5))
        plt.colorbar(img, ax=ax_img)
        ax_img.set_xlabel(r"$\delta$ (fractional lag index)")
        ax_img.set_ylabel(r"H (sub height index)")
        ax_img.set_title(f"$\\mathrm{{PSF}}^{{(H)}}_{{\\delta, {q}}}$")
        fractional_lag_indices = delta_min + np.arange(t.shape[0])
        order = 0
        while True:
            func = fractional_lag_indices**order
            contribution_by_height_indices = np.sum(t*func[:, np.newaxis], axis=0)
            if not np.all(contribution_by_height_indices == 0):
                axs_plot.scatter(np.arange(t.shape[1]), contribution_by_height_indices, marker="x", label=f"order={order}")
                break
            order += 1
        axs_plot.set_xlabel("H (sub height index)")
        axs_plot.set_ylabel("sum over $\delta$ axis")
        axs_plot.legend()
    plt.savefig("deconv_study/point_spread_funcs.pdf")
    plt.show()
    return r


g_barker_13 = [+1, +1, +1, +1, +1, -1, -1, +1, +1, -1, +1, -1, +1]

q_vals = [0, 1, 2, 3]
r = [calc_PSF(q, g_barker_13) for q in q_vals]
do_plot(r, q_vals, len(g_barker_13))

#%%

S = len(g_barker_13)
res = []
for q in range(-2*(S - 1), 2*(S - 1) + 1):
    p = calc_PSF(q, g_barker_13)
    res.append(p)
    plt.imshow(p.T, interpolation="None", origin="lower",)
    plt.show()

# verifies the symmetry
q_off = len(res)//2
for q in range(2*(S - 1) + 1):
    assert np.all(np.flip(res[q_off + q], 1) == res[q_off - q])




