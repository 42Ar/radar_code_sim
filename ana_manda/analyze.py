#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 18:07:26 2022

@author: frank
"""

import bz2
import numpy as np
from glob import glob
from scipy.io import loadmat
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from datetime import datetime
import os

import sys
sys.path.append("/home/frank/study/special_curriculum/isr_sim")
import consts
import ionosphere
import ACF
import config


# TX starts at 73 us
# 2.4 us baud length
# 1.2 us per sample
# 1.5 ms between pulses

# integration time 4800000
# loops 25

# channel store order 2 1 5 4
# CH2, CH5: 128 send samples + 10 calibration samples => vec_len=138
# CH1, CH4: 942 recieve samples => 942*128 => vec_len => 120576

# CH2         CH1         CH5         CH4
# 128*128*2 + 120576*25 + 128*128*2 + 120576*25 = 6094336 = file_size

# apparently sub_vec_len*nr_loops*res_mult is the blob size in the file, if send_raw=1
# vec_len represents the actual number of values (complex floats) "produced" by the tlan file
# apparently the order in the file is the order in which they are stored in the file

# it seems that the code is transmitted twice and then this is repeated 25 times
# what does nr_rep really mean? (signals are very similiar but not exactly the same)


# AD1L connects AD1 (one antenna half) to the left channel board group (1, 2, 3)
# AD2L connects AD2 (other antenna half) to the right channel board group (4, 5, 6)


# SP uses CH1, CH2, CH4, CH5 layout as above

# CP uses CH1, CH4 with
# CH1         CH1         CH4         CH4
# 128*128*2 + 120576*25 + 128*128*2 + 120576*25 = 6094336 = file_size

#code_info_file = '/home/frank/study/special_curriculum/manda_zenith_4.00v_SW@vhf_information/20180418/114824/manda-v.tlan.bz2'
#code_info_file = '/home/frank/study/special_curriculum/manda_zenith_4.00v_CP@vhf_information/20210304/120235/manda-v.tlan.bz2'
code_info_file = '/home/frank/study/special_curriculum/manda_zenith_4.00v_SP@vhf_information/20170429/213056/manda-v.tlan.bz2'
exp_file = bz2.open(code_info_file)

code = []
c, RFON_time, lastphase, rel_time = None, None, None, None, 
for line in exp_file:
    line = line.split(b"%")[0]
    if line.strip() == b"":
        continue
    splitted = line.split(b"\t")
    if splitted[0] != b"AT":
        continue
    time = float(splitted[1])
    if c is not None and (b"PHA0" in line or b"PHA180" in line or b"RFOFF" in line):
        delta = time - rel_time
        bits = round(delta/2.4)
        assert bits > 0 and abs(delta - bits*2.4) < 1e-10, "unexpected baud length"
        c += [lastphase]*bits
        rel_time = time
    if b"RFOFF" in line:
        assert c is not None, "RFOFF while RF was not on"
        assert len(c) == 61, "unexpected code length"
        code.append(c)
        c, rel_time = None, None
    if b"PHA0" in line:
        lastphase = 1
    elif b"PHA180" in line:
        lastphase = -1
    if b"RFON" in line:
        assert c is None, "RFON while RF was altready on"
        assert RFON_time is None or RFON_time == time, "inconsistent RFON_time"
        RFON_time = time
        rel_time = time
        c = []
code = np.array(code)
assert len(code) == 128, "unexpected subcycle count"
assert RFON_time == 73, "unexpected RFON_time"

#%%

def check_code():
    for k1 in range(code.shape[1]):
        print(k1)
        for k2 in range(code.shape[1]):
            for i1 in range(code.shape[1]):
                for i2 in range(code.shape[1]):
                    if k1 != k2 and i1 != k1 and i2 != k2 and abs(i1 - i2 - (k1 - k2)) <= 1:
                        r = sum(code[c, k1]*code[c, k2]*code[c, i1]*code[c, i2] for c in range(len(code)))
                        if r != 0:
                            print(f"code error {k1} {k2} {i1} {i2}: {r}")
check_code()

#%%


#data_folder = '/home/frank/study/special_curriculum/manda_zenith_4.00v_CP@vhf/20210304_12/*'
#data_folder = '/home/frank/study/special_curriculum/manda_zenith_4.00v_NO@vhf/20200706_07/*'
#data_folder = '/home/frank/study/special_curriculum/manda_zenith_4.00v_SP@vhf/20170430_00/*'
data_folder = '/home/frank/study/special_curriculum/manda_zenith_4.00v_SW@vhf/20180418_12/*'
infiles = glob(data_folder)
infiles.sort()
file_data = []
P = []
for i, file in enumerate(infiles[100:140]):
    print(f"reading {i}:", file)
    file = loadmat(bz2.open(file))
    params = file['d_parbl'][0]
    year = int(params[0])
    month = int(params[1])
    day = int(params[2])
    hour = int(params[3])
    minute = int(params[4])
    second = params[5]
    power = params[7]
    P.append(power)
    elevation_deg = params[8]
    az_deg = params[9]
    print(year, month, day, hour, minute, second, "P: ",
          power*1e-6, "MW", elevation_deg, az_deg)
    m = file['d_raw'][:, 0]
    assert 128*128*2 + 120576*25 + 128*128*2 + 120576*25 == len(m)
    assert az_deg == 90 and elevation_deg == 90
    file_data.append(m)

#%%

P = np.array(P)
min_power = 1.1e6
config.time = datetime(min(2020, year), month, day, hour, minute)  # using previous
config.P = np.mean(P[P > min_power])
print(f"average power {config.P*1e-6:1.3g} MW")
out_folder = config.time.isoformat()
try:
    os.mkdir(out_folder)
except FileExistsError:
    print("overwriting previous plots")
    

#%%

offset = 0  # 128*128*2 + 120576*25 the absolute value seems really small for second antenna
codenr = 0
m = file_data[0][offset + codenr*128:]
tx_delay_us = 5.2 # time offset between measurement start and start of tx
a = np.empty(61*2)
a[::2] = np.arange(62)[:-1]*2.4
a[1::2] = np.arange(62)[1:]*2.4
plt.plot(np.arange(128)*1.2 - tx_delay_us, np.angle(m[:128]), label="measured")
plt.plot(a, np.pi*(1 - np.repeat(code[codenr%128], 2))/2, label="transmitted")
plt.xlabel("time in us")
plt.ylabel("phase in radians")
plt.title(f"transmission of manda code {codenr}")
plt.legend()
plt.savefig(f"{out_folder}/offset.pdf", bbox_inches='tight')
plt.show()

#%%

# todo: proper filtering for mountains required

antenna_offs = [128*128*2, 128*128*2 + 120576*25 + 128*128*2][:1]
rx_tx_start = 73.6e-6 # when we start recieving the tx beam
rx_start = 343e-6 # when the actual tx start
ipp = rx_start - rx_tx_start - tx_delay_us*1e-6 # time between start of tx an begin of rx
mountain_dist = consts.c*(ipp + np.arange(942)*2.4e-6)/2
off = 150

#%%

s, e = off, off + 200
for off_in_buffer in antenna_offs:
    for sc in range(128):
        ss = []
        for fi, (m, power) in enumerate(zip(file_data, P)):
            for i in range(25):
                off = off_in_buffer + i*942*128 + sc*942
                ss.append(m[off:off + 942])
        mean = np.mean(ss, axis=0)
        for fi, (m, power) in enumerate(zip(file_data, P)):
            for i in range(25):
                off = off_in_buffer + i*942*128 + sc*942
                m[off:off + 942] -= mean

#%%

from scipy.signal import deconvolve

m = file_data[0]
off_in_buffer = antenna_offs[0]
data = m[off_in_buffer:off_in_buffer + 942]
r, res = deconvolve(data, code[0])
plt.plot(res[:200])
plt.show()

#%%

lags = [1, 40]
filt_by_lag = []
height_steps = 942//2 - 61 + 1
noise_analyze = False
noise_analyze_max_power = 0
for lag in lags:
    filt = []
    for fi, (m, power) in enumerate(zip(file_data, P)):
        if noise_analyze:
            if power > noise_analyze_max_power:
                continue
        else:
            if power < min_power:
                continue
        print(f"{lag}: {fi}/{len(file_data)}")
        for i in range(25):
            s = np.zeros(height_steps, dtype=np.complex128)
            for off_in_buffer in antenna_offs:
                for sc in range(128):
                    off = off_in_buffer + i*942*128 + sc*942
                    c = code[sc%128]
                    data = (m[off:off + 942:2] + m[off+1:off + 942:2])/2
                    corr = np.conj(data[lag:])*data[:942//2 - lag]
                    for k in range(61 - lag):
                        s += c[k]*c[k + lag]*corr[k:942//2 - 61 + 1 + k]
            filt.append(s/(len(antenna_offs)*128*(61 - lag)))
    filt_by_lag.append(np.array(filt))

#%%

total_height_steps = 942//2 + 61 - 1

def calc_expect_mm(R_by_h, c, j1, j2):
    assert j2 - j1 > 0
    # case h_diff = 0
    i1 = np.arange(code.shape[1] - (j2 - j1))
    i2 = i1 + j2 - j1
    r = np.sum(code[c, i1]*code[c, i2]*R_by_h[j1 - i1 + 61 - 1, np.abs(i2 - i1)])
    # case h_diff = 1 (half overlapping range gates)
    i1 = np.arange(code.shape[1] - (j2 - j1) + 1)
    i2 = i1 + j2 - j1 - 1
    r += np.sum(code[c, i1]*code[c, i2]*R_by_h[j1 - i1 + 61 - 1, np.abs(i2 - i1)])/2
    # case h_diff = -1 (half overlapping range gates)
    i1 = np.arange(code.shape[1] - (j2 - j1) - 1)
    i2 = i1 + j2 - j1 + 1
    r += np.sum(code[c, i1]*code[c, i2]*R_by_h[j1 - i1 + 61 - 1, np.abs(i2 - i1)])/2
    # slow impl, but equivalent
    #for i1 in range(code.shape[1]):
    #    for i2 in range(code.shape[1]):
    #        h_diff = i1 - i2 - (j1 - j2)
    #        if h_diff == 0:
    #            r += code[c, i1]*code[c, i2]*R_by_h[j1 - i1 + 61 - 1, abs(i2 - i1)]
    #        elif abs(h_diff) == 1:
    #            r += code[c, i1]*code[c, i2]*R_by_h[j1 - i1 + 61 - 1, abs(i2 - i1)]/2
    return r

def calc_expect():
    R_by_h = []
    shift = 1
    for hi in range(total_height_steps):
        h = (hi - 61 + 1)*2.4e-6*consts.c/2 + dist[0] + shift*1.5e-3*consts.c/2
        if h > ionosphere_start and h < expect_calc_end_h:
            spec_params = ionosphere.calc_spec_params(h)
            t, R = ACF.calc_R(h, spec_params, consts.c*2.4e-6/2)
            R_by_h.append(interp1d(t, R, assume_sorted=True)(np.arange(code.shape[1])*2.4e-6))
        else:
            R_by_h.append(np.zeros(code.shape[1]))
    R_by_h = np.array(R_by_h)
    res_by_lag = []
    for lag in lags:
        res_by_h = []
        for h in range(0, height_steps, 5):
            print(f"lag: {lag}, height: {h}")
            r = 0
            for c in range(128):
                for k in range(61 - lag):
                    r += code[c, k]*code[c, k + lag]*calc_expect_mm(R_by_h, (c + 128 - shift)%128, k + h, k + lag + h)
            res_by_h.append(r/(128*(61 - lag)))
        res_by_lag.append(res_by_h)
    return res_by_lag

time_diff = ipp + np.arange(height_steps)*2.4e-6
dist = time_diff*consts.c/2

ionosphere_start = 65e3
ionosphere_end = 1000e3
expect_calc_end_h = 450e3
alt_start = 72e3
ana_alt_step = 1e3
plot_expected_bias = True
dist_ana = np.arange(alt_start, dist[-1], ana_alt_step)
ionosphere.precalc_ionosphere(ionosphere_start, ionosphere_end, 1e3)

ana_res_by_lag = [[] for lag in lags]
for h in dist_ana:
    spec_params = ionosphere.calc_spec_params(h)
    t, R = ACF.calc_R(h, spec_params, consts.c*2.4e-6)
    R = interp1d(t, R, assume_sorted=True)
    for ana_res, lag in zip(ana_res_by_lag, lags):
        ana_res.append(R(lag*2.4e-6)/2)
if plot_expected_bias:
    expect_res_by_lag = calc_expect()
else:
    expect_res_by_lag = [[]]*len(lags)
#%%

# ionosphere.precalc_ionosphere(ionosphere_start, ionosphere_end, 1e3)
h_plot = np.linspace(75e3, 700e3, 200)
power_ana_h_plot = np.array([ACF.calc_power_at(h, consts.c*2.4e-6)/2 for h in h_plot])
plt.plot(h_plot*1e-3, power_ana_h_plot)
for a in range(3):
    plt.axvline((dist_ana[0] + 1.5e-3*consts.c*a/2)*1e-3, color="black", lw=1)
    plt.axvline((dist_ana[-1] + 1.5e-3*consts.c*a/2)*1e-3)
plt.xlabel("altitude in km")
plt.ylabel("power reflected from range gate in W")
plt.savefig(f"{out_folder}/power_by_alt.pdf", bbox_inches='tight')
plt.show()


#%%

# d layer starts at 90 km
scale_to_power = 0.9e-18  # found by fitting thermal noise => assumes correct filter and noise temp
# according to .elan file: set Filter1 $FIR/b300d18.fir    ;# +-300 kHz filter for 1.2 usec sampling
B = 2*300e3  # total bandwidth of filter in use
P_n = B*consts.k*config.T

power_ana_summed = sum(np.array([ACF.calc_power_at(h + i*1.5e-3/2*consts.c, consts.c*2.4e-6)/2 for h in dist_ana]) for i in range(4))

color = ["blue", "orange", "green"]
mode = "all" # "zoom_front"
plot_lags = [1, 40]
f_peak_enhance = 1
for lag, ana_res, expect_res, filt, col in zip(lags, ana_res_by_lag, expect_res_by_lag, filt_by_lag, color):
    if lag not in plot_lags:
        continue
    filt = np.real(filt)
    measured_power = np.mean(filt, axis=0)*scale_to_power
    uncertainty = np.std(filt, axis=0)*scale_to_power/np.sqrt(len(filt))
    
    samples = len(antenna_offs)*128*(61 - lag)*len(filt)
    thermal_noise = P_n/np.sqrt(samples)
    self_clutter_noise = 61*power_ana_summed/np.sqrt(samples)
    plt.fill_between(dist_ana*1e-3, ana_res - thermal_noise - self_clutter_noise, ana_res + thermal_noise + self_clutter_noise, color=col, alpha=0.4, zorder=-10000)
    
    plt.scatter(dist*1e-3, measured_power, color=col, marker=".", label=f"data, lag={lag*2.4:.2g}us")
    plt.plot(dist_ana*1e-3, ana_res, lw=1, color=col, zorder=100, label=f"expectation, lag={lag*2.4:.2g}us")
    
    if plot_expected_bias:
        plt.plot(dist[::5]*1e-3, f_peak_enhance*np.array(expect_res), lw=1, color=col, ls="-.", zorder=100)
    
    if mode == "zoom_front":
        chi2_mask = np.all([dist > 72e3, dist < 80e3], axis=0)
        chi2 = np.sum(measured_power[chi2_mask]**2/interp1d(dist_ana, thermal_noise + self_clutter_noise)(dist[chi2_mask])**2)
        print(chi2/np.sum(chi2_mask))
if mode == "zoom_front":
    plt.xlim(dist_ana[0]*1e-3, 90)
    plt.ylim(-0.1e-17, 0.1e-17)
elif mode == "zoom_back":
    plt.xlim(120, dist_ana[-1]*1e-3)
    plt.ylim(0.1e-17, 0.6e-17)
else:
    plt.xlim(dist_ana[0]*1e-3, dist_ana[-1]*1e-3)
    plt.ylim(0, np.max(ana_res)*1.6)
plt.title(f"integration time: {128*len(filt)*1.5e-3/60:3.2g} min, P={config.P*1e-6:.2g}MW, \n{config.time.isoformat()}")
plt.xlabel("altitude in km")
plt.ylabel("ACF in W")
plt.legend()
plt.grid()
plt.savefig(f"{out_folder}/mean_{mode}.pdf", bbox_inches='tight')
plt.show()

#%%

upper_ylim, lower_ylim = 0, 1e20
for lag, ana_res, filt, col in zip(lags, ana_res_by_lag, filt_by_lag, color):
    if lag not in plot_lags:
        continue
    samples = len(antenna_offs)*128*(61 - lag)*len(filt)
    thermal_noise = P_n/np.sqrt(samples)
    self_clutter_noise = 61*power_ana_summed/np.sqrt(samples)
    plt.axhline(thermal_noise, color=col, ls=":", label=f"thermal noise, lag={lag*2.4:.2g}us")
    plt.plot(dist_ana*1e-3, thermal_noise + self_clutter_noise, color=col, ls="-.", label=f"total expected noise, lag={lag*2.4:.2g}us")
    plt.plot(dist*1e-3, np.std(filt, axis=0)*scale_to_power/np.sqrt(len(filt)), color=col, label=f"observed noise, lag={lag*2.4:.2g}us")
    upper_ylim = max(upper_ylim, thermal_noise + self_clutter_noise[-1])
    lower_ylim = min(lower_ylim, thermal_noise)

    
plt.xlim(dist_ana[0]*1e-3, dist_ana[-1]*1e-3)
plt.ylim(lower_ylim - (upper_ylim - lower_ylim)*0.2, upper_ylim + (upper_ylim - lower_ylim)*2)
plt.xlabel("altitude in km")
plt.ylabel("standard deviation in W")
plt.title(f"integration time: {128*len(filt)*1.5e-3/60:3.2g} min, P={config.P*1e-6:.2g}MW, \n{config.time.isoformat()}")
plt.legend()
plt.savefig(f"{out_folder}/std.pdf", bbox_inches='tight')
plt.show()
