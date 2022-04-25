# -*- coding: utf-8 -*-

import sys
sys.path.append("/home/frank/study/special_curriculum/isr_sim")
sys.path.append("..")
import consts
import code_utils

lag_step = 1e-3
max_duty_cycle = 0.125
max_height = 800e3
plot_altitudes = True
N = 8
M = 6
loops = 25
c = [-1, +1, -1, +1, +1, +1, +1, +1, +1, +1, +1, -1, -1, +1, -1, -1, -1, -1, +1, -1, +1, +1, -1, -1, -1, +1, +1, -1, -1, -1, -1, +1, -1, -1, -1, +1, -1, -1, +1, +1, -1, +1, -1, -1, +1, -1, -1, +1]
g = [1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1]

def to_phase(i):
    if i == 1:
        return "PHA0"
    if i == -1:
        return "PHA180"
    raise ValueError()

code_utils.full_code_check(c, N, M)
code_utils.check_good_modulation_code(g)
file = open("/home/frank/study/radar_code/radar_code_sim/exp_prep/exp/code-v.tlan", "w")
pulse_len = lag_step*max_duty_cycle
baud_len_us = pulse_len*1e6/len(g)
baud_len_us = int(baud_len_us*10)/10
oversample_factor = 2
rx_sample_len_us = baud_len_us/oversample_factor
file.write(f"%% CODE(N={N}, M={M}, L={len(c)}): {c}\n")
file.write(f"%% MODULATION: {g}\n\n")
for ci, cc in enumerate(c):
    file.write("%%%%%%%%%%%%%%%%%%%%\n")
    file.write(f"%% SUBCYCLE {ci + 1}\n")
    file.write("%%%%%%%%%%%%%%%%%%%%\n")
    file.write(f"SETTCR	{ci*lag_step*1e6:.1f}\n")
    if ci == 0:
        file.write("AT	1.1	CHQPULS,RXSYNC,NCOSEL0,AD1L,AD2R,STFIR\n")
    file.write("AT	2	RXPROT,LOPROT\n")
    file.write("AT	32.9	BEAMON,F5\n")
    cur_phase = cc*g[0]
    tx_start_us = 73
    file.write("%% RF TRANSMISSION\n")
    file.write(f"AT\t{tx_start_us:.1f}\tCH2,CH5,RFON,{to_phase(cur_phase)}\n")
    for cj, gg in enumerate(g[1:]):
        if cc*gg != cur_phase:
            cur_phase = cc*gg
            file.write(f"AT\t{tx_start_us + (cj + 1)*baud_len_us:.1f}\t{to_phase(cur_phase)}\n")
    tx_end_us = tx_start_us + len(g)*baud_len_us
    file.write(f"AT\t{tx_end_us:.1f}\tRFOFF,PHA0,BEAMOFF\n")
    file.write(f"AT\t{tx_end_us + 7.2:.1f}\tALLOFF\n")
    file.write(f"AT\t{tx_end_us + 80:.1f}\tRXPOFF\n")
    file.write(f"AT\t{tx_end_us + 100:.1f}\tLOPOFF\n")
    file.write("%% SIGNAL RECEPTION\n")
    rx_start_us = tx_end_us + 123.6
    file.write(f"AT\t{rx_start_us:.1f}\tCH1,CH4\n")
    buffer_off_calib_us = 11.6
    calib_samples = 10
    samples = int((lag_step*1e6 - rx_start_us - buffer_off_calib_us - calib_samples*rx_sample_len_us)/rx_sample_len_us)
    if ci == 0:
        print(f"samples per subcycle: {samples}")
    rx_end_us = rx_start_us + rx_sample_len_us*samples
    if ci%2 == 0:
        file.write(f"AT\t{rx_end_us}\tALLOFF,CALON\n")
        file.write(f"AT\t{rx_end_us + buffer_off_calib_us}\tCH2,CH5\n")
        file.write(f"AT\t{rx_end_us + buffer_off_calib_us + calib_samples*rx_sample_len_us}\tALLOFF,CALOFF\n")
    else:
        file.write(f"AT\t{rx_end_us}\tALLOFF\n")
        file.write(f"AT\t{rx_end_us + buffer_off_calib_us}\tCH2,CH5\n")
        file.write(f"AT\t{rx_end_us + buffer_off_calib_us + calib_samples*rx_sample_len_us}\tALLOFF\n")
    file.write("\n")
file.write(f"AT\t{lag_step*len(c)*1e6:.1f}\tREP\n\n")
file.close()
range_gate_height = baud_len_us*consts.c*1e-6
print(f"resulting pulse length: {baud_len_us*len(g):.5g}us")
print(f"resulting duty cycle: {baud_len_us*len(g)/lag_step*1e-6*100:.5g}%")
print(f"resulting height resolution: {range_gate_height*1e-3:.5g}km")
print(f"resulting baud length: {baud_len_us:.5g}us")
max_height = (N*lag_step + rx_start_us*1e-6 - tx_end_us*1e-6)*consts.c/2
print(f"resulting max height: {max_height*1e-3:.5g}km")
print(f"experiment time {loops*lag_step*len(c):.2g}s")

if plot_altitudes:
    import matplotlib.pyplot as plt
    import numpy as np
    x, y = [], []
    for i in range(N):
        h_0 = (i*lag_step + rx_start_us*1e-6 - tx_end_us*1e-6)*consts.c/2
        h_1 = (i*lag_step + rx_start_us*1e-6 - tx_start_us*1e-6)*consts.c/2
        h_2 = (i*lag_step + rx_end_us*1e-6 - tx_end_us*1e-6)*consts.c/2
        h_3 = (i*lag_step + rx_end_us*1e-6 - tx_start_us*1e-6)*consts.c/2
        print(f"region {i + 1}: {h_1*1e-3:.4g}km - {h_2*1e-3:.4g}km")
        y += [h_0, h_1, h_2, h_3]
        x += [0, 1, 1, 0]
    x.append(0)
    y.append(max_height)
    plt.plot(x, np.array(y)*1e-3)
    plt.axhline(max_height*1e-3, color="black")
    plt.show()
