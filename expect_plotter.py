# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from calc_expect import calc_r, setup, index_to_alt, default_param_provider, calc_expect_on_alt
from deconv_study.psf_calculator import calc_PSF_array
from datetime import datetime


exp_info = {
    "g": [+1, +1, +1, +1, +1, -1, -1, +1, +1, -1, +1, -1, +1],
    "W": 125,
    "type": "physical",
    "tau_IPP": 0.5e-3,
    "tau": 6e-6,
    "tau_d": 123.6e-6,
    "M": 10
}

h = 0
PSF_array = calc_PSF_array(exp_info["g"])
acf_len = 50
M = 10
w_start = 70
w_stop = 70
delta_w_step = 9
setup(datetime(2018, 5, 5, 12), 1e6, 65e3, 200e3, 0.5e3)
r = calc_r(h, np.arange(acf_len), w_start, w_stop, exp_info, PSF_array,
           default_param_provider, False, True)
S = len(exp_info["g"])

#%%

Delta_vals = np.arange(20)
acf = calc_expect_on_alt(h, w_start + S - 1, Delta_vals, exp_info,
                         default_param_provider)[0][:, S//2]
plt.plot(Delta_vals, acf, marker="o", ls="None")

print(acf[0]*13**2*2, r[1, 0, 0], 6.77e-21*13**2*2, acf[0]/acf[1])

#%%

w_off = 0
for delta_w in range(0, 2*S, delta_w_step):
    plt.scatter(np.arange(1, acf_len), r[1:, w_off, delta_w], marker="o",
                label=f"$\\delta_w={delta_w}$", s=10)
alt = index_to_alt(exp_info, 0, w_start + w_off + S - 1)
plt.title(f"last-to-cutoff ratio {r[-1, w_off, 0]/r[M, w_off, 0]*100:.1f}% and power {r[0, w_off, 0]:.2g} W (for $\\delta_w=0$), alt: {alt/1e3:.1f} km")
plt.xlabel("lag index")
plt.ylabel("power in W")
plt.legend()
plt.grid()
plt.show()