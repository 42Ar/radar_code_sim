import numpy as np
from glob import glob
from scipy.io import loadmat
import matplotlib.pyplot as plt

from datetime import datetime, timedelta
import code_utils
import exp_io
from pathlib import Path
from analyze_barker_modulation.decoder import decode, filter_sub_pulse

import sys
sys.path.insert(0, "../../special_curriculum/isr_sim")
import ACF
import config
import consts
from ionosphere import precalc_ionosphere


loops = 50
folder = 'code_56'
data_folder = f'/data/barker_pulses/{folder}/*'
if folder == 'code_48':
    c = [-1, +1, -1, +1, +1, +1, +1, +1, +1, +1, +1, -1, -1, +1, -1, -1, -1, -1, +1, -1, +1, +1, -1, -1, -1, +1, +1, -1, -1, -1, -1, +1, -1, -1, -1, +1, -1, -1, +1, +1, -1, +1, -1, -1, +1, -1, -1, +1]
elif folder == 'code_56':
    N = 4
    M = 10
    c = [-1, -1, -1, -1, -1, -1, -1, -1, +1, +1, -1, +1, +1, +1, +1, -1, -1, -1, -1, +1, +1, -1, +1, +1, -1, -1, -1, +1, -1, +1, +1, -1, -1, +1, -1, +1, +1, -1, +1, +1, +1, -1, -1, +1, +1, +1, -1, +1, -1, +1, -1, +1, +1, +1, -1, +1]
g = [+1, +1, +1, +1, +1, -1, -1, +1, +1, -1, +1, -1, +1]
calib_samples = 10
baud_len = 6e-6
samples_per_subcycle = 125
samples_after_RFOFF = 10
sky_power_start = 70e3
min_power = 0.9e6
integration_time = 60*6
max_out_files = -1
lag_step = 1100e-6
max_filter_width = 5
max_filter_threshold = 3.8717891660142174e-14
lower_mean_filter_threshold = 4.519704281427461e-15
cut_around_max_filt = 10
rx_start_us = 274.6
tx_start_us = 73
tx_end_us = 151.0
B = 2*83e3  # total bandwidth in Hz
skip_pulses_after_integration = 0

n_codes = int(integration_time/(len(c)*lag_step))
ipp_n = len(c)*loops
calib_vec_n = len(g) + calib_samples + samples_after_RFOFF
calib_n = calib_vec_n*ipp_n
rx_n = samples_per_subcycle*ipp_n
ch = (0,)# calib_n + samples_per_subcycle*ipp_n)
n_filtered_samples = samples_per_subcycle - len(g) + 1
time_per_file = loops*len(c)*lag_step


def calibrate(data, T):
    calib_ch_data = data.reshape(ipp_n, calib_vec_n)
    calib_data = calib_ch_data[:, len(g) + calib_samples:]
    assert calib_data.shape[1] == calib_samples
    power_calib_on = np.mean(np.abs(calib_data[::2])**2)
    power_calib_off = np.mean(np.abs(calib_data[1::2])**2)
    sigma = np.sqrt(consts.k*B*T/power_calib_on)
    T_sys = T*power_calib_off/power_calib_on
    return sigma, T_sys


def h_to_i(h):    
    # it is (rx_start_us - tx_start_us) and not (rx_start_us - tx_end_us)
    # due to the sub pulse filtering (but still just approximately; PSF
    # not taken into account)
    dt = h*2/consts.c - (rx_start_us - tx_start_us)*1e-6
    return round(dt/baud_len)


def load_file(fname):
    file = loadmat(fname)
    data = file['d_raw'][:, 0]
    assert (calib_vec_n + samples_per_subcycle)*len(c)*loops*2 == len(data)
    params = file['d_parbl'][0]
    return data, params


def calc_T_sys_power_off(power_off_file):
    data, params = load_file(power_off_file)
    power = params[7]
    T = params[20]
    assert power == 0
    T_sys = 0
    for i, ch_off in enumerate(ch):
        sigma, T_sys_ch = calibrate(data[ch_off:ch_off + calib_n], T)
        T_sys += T_sys_ch
    return T_sys


class IntegrationResult:
    def __init__(self, first_date, index, start, T_sys):
        self.first_date = first_date
        self.index = index
        self.start = start
        self.T_sys = T_sys
        self.n_codes = 0
        self.P = 0
        self.decoded = np.zeros((N, M, n_filtered_samples), dtype=np.complex128)
        self.power_profile = np.zeros(n_filtered_samples)
    
    def write(self):  
        exp_info = {
            "year": self.first_date.year,
            "month": self.first_date.month,
            "day": self.first_date.day,
            "hour": self.first_date.hour,
            "minute": self.first_date.minute,
            "second": self.first_date.second,
            "start": self.start,
            "end": self.end,
            "index": self.index,
            "g": g,
            "c": c,
            "N": N,
            "M": M,
            "W": samples_per_subcycle,
            "tau": baud_len,
            "tau_IPP": lag_step,
            "tau_d": (rx_start_us - tx_end_us)*1e-6,
            "B": B,
            "pulses": self.n_codes*len(c),
            "P_mean": self.P,
            "T_sys": self.T_sys
        }
        exp_io.save(exp_info, self.decoded)


code_utils.full_code_check(c, N, M)
T_sys = calc_T_sys_power_off("/data/barker_pulses/code_56/10839651.mat")
print(f"T_sys: {T_sys:.0f} K")

#%%

infiles = glob(data_folder)
infiles.sort()
cc = np.tile(c, loops)
sky_power_start_i = h_to_i(sky_power_start)
res, sky_powers_mean, sky_powers_max, P, T_sys_vals, int_periods, files_processed = [], [[]]*len(ch), [[]]*len(ch), [], [], [], []
first_date, cur_res = None, None
assert max_filter_width%2 == 1
for i, fname in enumerate(infiles):
    data, params = load_file(fname)
    date = datetime(int(params[0]), int(params[1]), int(params[2]),
                    int(params[3]), int(params[4]), int(params[5]))
    if first_date is None:
        first_date = date
    else:
        assert abs((date - first_date).total_seconds() - i*time_per_file) < 1
    assert params[6] == time_per_file
    power = params[7]
    P.append(power)
    elevation_deg = params[8]
    az_deg = params[9]
    T = params[20]
    assert az_deg == elevation_deg
    print(f"{i:3d}: {Path(fname).stem}, {date}, P: {power*1e-6:.2f} MW, {elevation_deg:2.0f}°, T: {T:.1f} K")
    if int(params[63]) != loops:
        print(f"WARN: discarding {fname}: wrong loop count {loops} != {int(params[63])}")
        continue
    if power < min_power:
        print(f"WARN: discarding {fname}: not enough power on klystron")
        continue
    assert az_deg == 90 and elevation_deg == 90
    assert (calib_vec_n + samples_per_subcycle)*len(c)*loops*2 == len(data)
    ds, filt = [], []
    discard_file = False
    for ch_i, ch_off in enumerate(ch):
        sigma, T_eff = calibrate(data[ch_off:ch_off + calib_n], T)
        m = sigma*data[ch_off + calib_n:ch_off + calib_n + rx_n].reshape(ipp_n, samples_per_subcycle)
        d = filter_sub_pulse(m, g)
        ds.append(d)
        sky_power = np.abs(d[:, sky_power_start_i:])**2
        sky_power_mean = np.mean(sky_power, 1)
        if np.mean(sky_power_mean) < lower_mean_filter_threshold:
            discard_file = True
        sky_powers_mean[ch_i].append(sky_power_mean)
        sky_power_max = np.max(sky_power, 1)        
        sky_powers_max[ch_i].append(sky_power_max)
        filt.append(np.convolve(sky_power_max, np.ones(max_filter_width)/max_filter_width, "valid") > max_filter_threshold)
    files_processed.append(i)
    if discard_file:
        print(f"WARN: discarding {fname}: unusual low recieve power")
        continue
    d = np.sum(ds, 0)
    filt = np.any(filt, 0)
    filt_offset = np.argmax(~filt)  # first not filtered pulse
    while not filt[filt_offset] and len(res) != max_out_files:
        filt_end = np.argmax(filt[filt_offset:]) + filt_offset
        eof = filt_end == filt_offset
        if eof:
            end = len(c)*loops
        else:
            end = filt_end + max_filter_width//2 - cut_around_max_filt
        assert end <= len(c)*loops
        #  note: end not inclusive
        end -= N + M - 1  # need this space to decode
        offset = filt_offset + max_filter_width//2 + cut_around_max_filt
        while len(res) != max_out_files:
            n_usable_codes = (end - int(offset))//len(c)
            if n_usable_codes <= 0:
                break
            if cur_res is None:
                time_offset = (i*loops*len(c) + offset)*lag_step
                cur_res = IntegrationResult(first_date, len(res), time_offset, T_sys)
            remaining_codes = n_codes - cur_res.n_codes
            codes_to_use = min(remaining_codes, n_usable_codes)
            cur_res.P += power*codes_to_use
            w = codes_to_use*len(c)
            cur_res.decoded += decode(d, offset, w, c, N, M)*w
            cur_res.n_codes += int(codes_to_use)  # avoid np.int64
            cur_res.power_profile += np.sum(np.abs(d[offset:offset + codes_to_use*len(c)])**2, 0)
            int_periods.append([i*loops*len(c) + offset, i*loops*len(c) + offset + codes_to_use*len(c) + N + M - 1])
            offset += w
            if cur_res.n_codes == n_codes:
                cur_res.power_profile /= n_codes*len(c)
                cur_res.decoded /= n_codes*len(c)
                cur_res.P /= n_codes
                cur_res.end = (i*loops*len(c) + offset + codes_to_use*len(c))*lag_step
                print(f"writing integration period {cur_res.index}")
                cur_res.write()
                res.append(cur_res)
                cur_res = None
                offset += skip_pulses_after_integration
        if eof:
            break
        filt_offset = np.argmax(~filt[filt_end:]) + filt_end
    if len(res) == max_out_files:
        break
total_power_profile = np.array([r.power_profile for r in res]).mean(0)
total_decoded = np.array([np.real(r.decoded) for r in res]).mean(0)
total_decoded_std = np.array([np.real(r.decoded) for r in res]).std(0)/np.sqrt(len(res))
sky_powers_mean = np.array(sky_powers_mean)
sky_powers_max = np.array(sky_powers_max)
int_periods = np.array(int_periods)
files_processed = np.array(files_processed)
P_mean = np.array([r.P for r in res]).mean()


#%%

w = 1
filter_calc_start = 1000
filter_calc_end = 20000
filter_calc_ch = 0
lower_filter_calc_nfiles = 5
all_filt = np.convolve(sky_powers_max[filter_calc_ch].ravel()[filter_calc_start:filter_calc_end + 1], np.ones(max_filter_width)/max_filter_width, "valid")
upper = np.max(all_filt)
m = sky_powers_mean[filter_calc_ch].mean(1)[:lower_filter_calc_nfiles].mean()
s = sky_powers_mean[filter_calc_ch].mean(1)[:lower_filter_calc_nfiles].std()
lower = m - 5*s
for chi_i in range(len(ch)):
    x = np.ravel([range(i*len(c)*loops, (i + 1)*len(c)*loops) for i in files_processed])
    plt.scatter(x, sky_powers_max[chi_i].ravel(), label=f"channel {chi_i + 1} max powers", s=3, marker="o")
    plt.scatter(x, sky_powers_mean[chi_i].ravel(), label=f"channel {chi_i + 1} mean powers", s=3, marker="^")
    x = np.ravel([range(i*len(c)*loops + max_filter_width//2, (i + 1)*len(c)*loops - max_filter_width//2) for i in files_processed])
    y = [np.convolve(a, np.ones(max_filter_width)/max_filter_width, "valid") for a in sky_powers_max[chi_i]]
    plt.plot(x, np.array(y).ravel(), label=f"channel {chi_i + 1}, filter", marker="X")
    x = (files_processed + 0.5)*len(c)*loops
    y = sky_powers_mean[chi_i].mean(1)
    plt.plot(x, y, marker="o", color="black")
for p in int_periods:
    plt.axvspan(*p, alpha=0.3, color='grey')
plt.title("filter diagnostics")
plt.xlabel("rx pulse index")
plt.ylabel("rx power in W")
plt.ylim(lower - (upper - lower)/8, upper + (upper - lower)/8)
plt.xlim(np.min(x), np.max(x))
plt.axhline(upper, color="red")
plt.axhline(lower, color="red")
plt.legend(loc="upper right")
plt.grid()
plt.show()

#%%

def plot_sky_power_hist():
    res = []
    for i, powers in enumerate(sky_powers_mean):
        upper = np.quantile(powers, 0.95)
        lower = np.quantile(powers, 0.05)
        mean_powers_filt = powers[(powers > lower) & (powers < upper)]
        res.append(mean_powers_filt.mean())
        plt.hist(mean_powers_filt, bins=150, label=f"channel {i}", alpha=0.8)
    diff = res[0]/res[1] - 1
    plt.title(f"histogram of all sky powers: average diff {diff*100:.1f}%")
    plt.xlabel("sky power in W")
    plt.ylabel("count")
    plt.yscale("log")
    plt.legend()
    plt.show()

plot_sky_power_hist()


#%%

plot_imag = False
plot_start_h = 70e3

h_end = consts.c*(samples_per_subcycle*baud_len + (rx_start_us - tx_start_us)*1e-6)/2
plot_start_h_i = h_to_i(plot_start_h)
heights = np.linspace(plot_start_h, h_end, n_filtered_samples - plot_start_h_i)
config.P = P_mean
config.time = res[0].first_date - timedelta(365*2, 0)
precalc_ionosphere(70e3, 1000e3, 1e3)
pp_model = np.array([[ACF.calc_power_at(h + consts.c*lag_step*k/2,
                                        baud_len*consts.c/3)
                      for h in heights]
                     for k in range(5)])

for h in [0]:
    for Delta in range(1, M + 1):
        plt.errorbar(heights*1e-3, total_decoded[h, Delta - 1, plot_start_h_i:], total_decoded_std[h, Delta - 1, plot_start_h_i:], label=f"lag: {Delta}", marker="o", ls="--", capsize=3)
        if plot_imag:
            plt.scatter(heights*1e-3, np.imag(total_decoded[h, Delta - 1, plot_start_h_i:]), label=f"lag: {Delta}", ls="--")

plt.plot(heights*1e-3, total_power_profile[plot_start_h_i:] - total_power_profile[plot_start_h_i:].min(), label="power profile", color="black", ls="--")
plt.plot(heights*1e-3, consts.k*T_sys*B*len(g) + pp_model.sum(0)*len(g)**2, label="power profile model", c="r")
plt.axhline(consts.k*T_sys*B*len(g), label="noise floor", color="black", ls=":")

plt.legend()
plt.grid()
plt.xlim(heights[0]*1e-3, heights[-1]*1e-3)
plt.title(f"decoded lags from the 56 bit code, match filtered, integration time: {n_codes*len(c)*lag_step*len(res)/60:.0f} min")
plt.xlabel("altitude in km")
plt.ylabel("Re(ACF(lag)) in W")
plt.show()