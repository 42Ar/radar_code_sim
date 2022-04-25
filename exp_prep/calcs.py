#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 16:08:20 2022

@author: frank
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

import sys
sys.path.append("/home/frank/study/special_curriculum/isr_sim")
import consts
import ionosphere
import ACF
import config

h = 80e3
baud_length = 2.4e-6
half_filter_width_manda = 300e3 # Hz
config.P = 1.1e6
T = 60*60*1
config.time = datetime(2020, 5, 5, 10)
height_avg = 20
print(f"height resolution {height_avg*baud_length*consts.c*1e-3/2:.2g}km")

print(f"average power manda {config.P*baud_length*61*1e-3/1.5e-3:.4g}kW")

ionosphere.precalc_ionosphere(75e3, 1000e3, 1e3)

lags_manda = baud_length*np.arange(1, 61)
samples_manda = height_avg*2*np.flip(np.arange(1, 61))*(T/1.5e-3)  # factor two because we have two recievers
P_n_manda = 2*consts.k*half_filter_width_manda*config.T
P_clutter = sum(ACF.calc_power_at(h + i*1.5e-3*consts.c/2, consts.c*baud_length/2) for i in range(5))
P_upper = sum(ACF.calc_power_at(h + i*1.5e-3*consts.c/2, consts.c*baud_length/2) for i in range(1, 5))
expected_manda_bias = P_upper/61
noise_manda = (P_n_manda + 61*P_clutter)/np.sqrt(samples_manda)

spec_params = ionosphere.calc_spec_params(h)
t, R = ACF.calc_R(h, spec_params, consts.c*baud_length/2)
plt.plot(t*1e3, R)
plt.plot(lags_manda*1e3, noise_manda, label="expected noise")
plt.axvline(60*baud_length*1e3, label="max short lag manda")
plt.axhline(expected_manda_bias, label="expected bias", color="black")
plt.xlim(0, 1)
plt.ylim(0, 2e-19)
plt.axvline()
plt.xlabel("ms")
plt.ylabel("ACF")
plt.legend()
plt.show()

#%%

max_height = 800e3
hvals = np.linspace(80e3, 900e3, 100)
power = [ACF.calc_power_at(t, consts.c*baud_length/2) for t in hvals]
plt.plot(hvals*1e-3, power)
plt.ylim(np.min(power), np.max(power))
plt.xlabel("km")
plt.ylabel("power")
plt.yscale("log")
plt.axhline(R[0])
plt.axvline(max_height*1e-3)
plt.show()





