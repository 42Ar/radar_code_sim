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


# from /home/frank/study/special_curriculum/isr_sim
import sys
sys.path.append("/home/frank/study/special_curriculum/isr_sim")
import consts
import ionosphere
import ACF


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


code_info_file = '/data/manda_zenith_4.00v_CP@vhf_information/20210811/075349/manda-v.tlan.bz2'
exp_file = bz2.open(code_info_file)

c, code = [], []
for line in exp_file:
    if b"%" in line and b"PHA0" in line:
        c += [1]*line.split(b"%")[1].count(b"+")
    elif b"%" in line and b"PHA180" in line:
        c += [-1]*line.split(b"%")[1].count(b"-")
    elif b"SIGNAL RECEPTION" in line:
        assert len(c) == 61
        code.append(c)
        c = []
code = np.array(code)
assert len(code) == 128

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

data_folder = '/data/manda_zenith_4.00v_CP@vhf/20210811_12/*'
infiles = glob(data_folder)
infiles.sort()
file_data = []
for file in infiles[:30]:
    print("reading", file)
    file = loadmat(bz2.open(file))
    params = file['d_parbl'][0]
    year = int(params[0])
    month = int(params[1])
    day = int(params[2])
    hour = int(params[3])
    minute = int(params[4])
    second = params[5]
    power = params[7]
    elevation_deg = params[8]
    az_deg = params[9]
    print(year, month, day, hour, minute, second, "P: ",
          power*1e-6, "MW", elevation_deg, az_deg)
    m = file['d_raw'][:, 0]
    assert 128*128*2 + 120576*25 + 128*128*2 + 120576*25 == len(m)
    file_data.append(m)

#%%

tx_delay_us = 5.2 # time offset between measurement start and start of tx
a = np.empty(61*2)
a[::2] = np.arange(62)[:-1]*2.4
a[1::2] = np.arange(62)[1:]*2.4
plt.plot(np.arange(128)*1.2 - tx_delay_us, np.angle(m[:128]), label="measured")
plt.plot(a, np.pi*(1 - np.repeat(code[0], 2))/2, label="transmitted")
plt.xlabel("time in us")
plt.ylabel("phase in radians")
plt.title("transmission of first manda code")
plt.legend()
plt.show()

#%%

off_in_buffer_1 = 128*128*2
off_in_buffer_2 = 128*128*2 + 120576*25 + 128*128*2
lag = 2
filt = []
for fi, m in enumerate(file_data):
    print(fi, len(file_data))
    for i in range(25):
        s = np.zeros(942 - 2*61 + 1, dtype=np.complex128)
        for off_in_buffer in (off_in_buffer_1, off_in_buffer_2):
            for sc in range(128):
                off = off_in_buffer + i*942*128 + sc*942
                c = np.repeat(code[sc%128], 2)
                data = m[off:off + 942]
                corr = np.conj(data[lag:])*data[:942 - lag]
                for k in range(61*2 - lag):
                    s += c[k]*c[k + lag]*corr[k:942 - 2*61 + 1 + k]
        filt.append(s/(2*(61*2 - lag)))
filt = np.array(filt)
r = np.mean(filt, axis=0)

#%%

rx_tx_start = 73.6e-6 # when we start recieving the tx beam
rx_start = 343e-6 # when the actual tx start

ipp = rx_start - rx_tx_start - tx_delay_us*1e-6 # time between start of tx an begin of rx
time_diff = ipp + np.arange(len(r))*1.2e-6
dist = time_diff*consts.c/2

alt_start = 72e3
ana_alt_step = 1e3
dist_ana = np.arange(alt_start, dist[-1], ana_alt_step)
ionosphere.precalc_ionosphere(dist_ana[0], dist_ana[-1] + ana_alt_step, ana_alt_step)
power_ana = [ACF.calc_power_at(h) for h in dist_ana]

#%%

plt.ylim(0, np.max(power_ana)*2)
plt.xlim(dist_ana[0]*1e-3, dist_ana[-1]*1e-3)
plt.plot(dist_ana*1e-3, power_ana)
plt.plot(dist*1e-3, np.real(r)*1.8e-20)
plt.xlabel("altitude in km")
plt.ylabel(f"ACF({lag*1.2} us) in W")
plt.show()


