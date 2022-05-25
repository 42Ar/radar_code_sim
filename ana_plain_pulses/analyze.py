import bz2
import numpy as np
from glob import glob
from scipy.io import loadmat
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from datetime import datetime
import os
from scipy import linalg
import numpy.random as rand

import sys
sys.path.append("/home/frank/study/special_curriculum/isr_sim")
import consts
import ionosphere
import ACF
import config


loops = 25
N = 3
M = 7
c = [-1, +1, -1, +1, +1, +1, +1, +1, +1, +1, +1, -1, -1, +1, -1, -1, -1, -1, +1, -1, +1, +1, -1, -1, -1, +1, +1, -1, -1, -1, -1, +1, -1, -1, -1, +1, -1, -1, +1, +1, -1, +1, -1, -1, +1, -1, -1, +1]
seed = 4178
g = list(rand.default_rng(seed).integers(0, 2, 66)*2 - 1)
calib_samples = 10
oversample_factor = 2
baud_len = 0.6e-6
samples_per_subcycle = 1216
samples_after_RFOFF = 10
data_folder = '/data/plain_pulses/*'
ana_min_h = 70e3
min_power = 0.9e6
mountain_echos_start_i = 0
mountain_echos_end_i = 30
max_sky_power = 240
min_sky_power = 80
min_calib_amp = 100
nfiles = int(60*10/1.23)
off = len(c)
w = (loops - 2)*len(c)
rx_start_us = 275.8
tx_end_us = 152.2
diagnostic = False

def h_to_i(h):    
    dt = h*2/consts.c - (rx_start_us - tx_end_us)*1e-6
    return round(dt/baud_len)

infiles = glob(data_folder)
infiles.sort()
selected_files = infiles[1000:1000 + nfiles]
P, power_stats = [], []
cc = np.tile(c, loops)
ana_min_h_i = h_to_i(ana_min_h)
ch1 = (len(g)*oversample_factor + calib_samples + samples_after_RFOFF)*len(c)*loops
ch2 = (2*len(g)*oversample_factor + 2*calib_samples + 2*samples_after_RFOFF + samples_per_subcycle)*len(c)*loops
res, calib_amps, sky_powers, ms, filt = [], [], [], [], []
power_profile = np.zeros(samples_per_subcycle)
for i, fname in enumerate(selected_files):
    print(f"reading {i}:", fname)
    file = loadmat(fname)
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
    data = file['d_raw'][:, 0]
    assert (len(g)*oversample_factor + calib_samples + samples_per_subcycle + samples_after_RFOFF)*len(c)*loops*2 == len(data)
    assert az_deg == 90 and elevation_deg == 90
    if power < min_power:
        print(f"discarding {f}: no enough power on klystron")
    assert power > 0.9e6
    file_res = np.zeros((N, M, samples_per_subcycle - ana_min_h_i), dtype=np.complex128)
    discard = False
    for channel_offset in (ch1, ):
        m = data[channel_offset:channel_offset + samples_per_subcycle*len(c)*loops].reshape(len(c)*loops, samples_per_subcycle)
        calib_amp = np.mean(np.abs(m[:, mountain_echos_start_i:mountain_echos_end_i]), axis=1)
        sky_power = np.mean(np.abs(m[:, ana_min_h_i:]), axis=1)
        calib_amps.append(calib_amp)
        sky_powers.append(sky_power)
        if diagnostic:
            ms.append(m)
        if discard or np.any(calib_amp < min_calib_amp) or np.any(sky_power < min_sky_power) or  np.any(sky_power > max_sky_power):
            discard = True
            continue
        power_profile += np.sum(np.abs(m), axis = 0)
        m = m[:, ana_min_h_i:]/sky_power[:, np.newaxis]
        for h in range(N):
            d1 = m[off + h:off + h + w]*cc[off:off + w, np.newaxis]
            for Delta in range(1, M + 1):
                d2 = m[off + h + Delta:off + h + Delta + w]*cc[off + Delta:off + Delta + w, np.newaxis]
                file_res[h, Delta - 1] += np.sum(np.conj(d2)*d1, axis=0)
    filt.append(discard)
    if discard:
        print(f"discarding {fname}: not enough mountain echos or to much sky power")
        continue
    res.append(file_res)
calib_amps = np.array(calib_amps).ravel()
sky_powers = np.array(sky_powers).ravel()

#%%

plt.scatter(range(len(sky_powers)), sky_powers)
plt.axhline(min_sky_power)
plt.axhline(max_sky_power)
plt.plot(np.array(filt).repeat(25*48)*np.mean(sky_powers), color="red")
plt.xlabel("ipp index")
plt.ylabel("total power from sky in ipp")
plt.show()

#%%

plt.hist(sky_power, bins=100)
plt.show()

#%%

plt.scatter(range(len(calib_amps)), calib_amps)
plt.axhline(min_calib_amp)
plt.plot(np.array(filt).repeat(25*48)*np.mean(calib_amps), color="red")

#%%

for spec in list(ms[0][84:84+4]) + list(ms[0][178:178+1]):
    plt.scatter(range(len(spec)), np.abs(spec), s=7)
    smooth_size = 66*2
    smoothed = np.convolve(np.abs(spec), np.ones(smooth_size)/smooth_size, "same")
    plt.plot(smoothed, lw=3)
plt.xlabel("proportional to altitude")
plt.ylabel("abolute squared value")
plt.title("unexplained behaviour in reciever")
plt.show()

#%%

mean = np.mean(res, axis=0)
err_real = np.std(np.real(res), axis=0)/np.sqrt(len(selected_files))
err_imag = np.std(np.imag(res), axis=0)/np.sqrt(len(selected_files))

h_end = consts.c*(samples_per_subcycle*baud_len + (rx_start_us - tx_end_us)*1e-6)/2
heights = np.linspace(ana_min_h, h_end, samples_per_subcycle - ana_min_h_i)

#smooth_size = 66*2
#for h in [0]:
#    for Delta in range(1, 7):
#        smoothed = np.real(np.convolve(mean[h, Delta - 1], np.ones(smooth_size)/smooth_size, "same"))
#        plt.plot(heights*1e-3, smoothed, label=f"lag: {Delta}")
plt.plot(np.linspace(consts.c*(rx_start_us - tx_end_us)*1e-6/2, h_end, samples_per_subcycle)*1e-3, power_profile)
plt.legend()
plt.xlabel("altitude in km")
plt.ylabel("Re(ACF)")
plt.show()

#%%


plt.plot(power_stats)




#%%



for h in [0, 1]:
    plt.errorbar(np.arange(1, 1 + M), np.real(mean[h]), err_real[h], label=f"h={h}")
    #plt.errorbar(np.arange(1, 1 + M), np.imag(mean[h]), err_imag[h], marker="x", label=f"h={h}")
plt.legend()
plt.grid()
plt.show()

#%%

res = np.zeros((N, M, samples_per_subcycle//2), dtype=np.complex128)
for data in file_data:


#%%

gp = np.flip(g)  # gamma'
W = samples_per_subcycle//2
S = len(g)
K = np.zeros((W, S + W))
for w in range(W):
    K[w, w] += 1
    K[w, w + 1] += 1
    for s in range(1, S):
        K[w, w + s] += 1 + 2*gp[s - 1]*gp[s]
        K[w, w + s + 1] += 1
u, s, v = linalg.svd(K)
trunc = 0
v_trunc = v[:W-trunc, :]
c = v_trunc.T @ np.diag(1/s[:W-trunc]**2) @ v_trunc


#for h in range(N):
#    for Delta in range(N):
h = 1
Delta = 0
r = c @ K.T @ res[h, Delta]

plt.plot(np.real(r[500:]))


#%%

import numpy.random as rand

S = 66  # number of bits
W = 1174//2  # number of samples
fractional_offset = S + W

#%%

K = np.zeros((W*(W + 1)//2, W*(2*S - 1)))
def set_K(w1, w2, delta, H, v):
    r = W*w2 + w1 - w2*(w2 + 1)//2
    if H < S:
        c = H**2 + (H - delta)
    elif H < W:
        # probably buggy
        c = S**2 + (2*S - 1)*(H - S - 1) + (S - delta)
    else:
        # probably buggy
        c = W*(2*S - 1) - S**2 + (H - W)**2 + (H - W - delta)
    K[r, c] = v
for w1 in range(W):
    for w2 in range(W):
        delta = w1 - w2
        for s in range(S - delta):
            v = gp[s]*gp[s + delta]

#%%

c_res = {}


#%%

def calc(gp):
    K = np.zeros((W, (W + S - 2)*2 + 2))
    for w in range(W):
        K[w, w] += 1
        K[w, w + 1] += 1
        for s in range(1, S):
            K[w, w + s] += 1
            K[w, w + s + 1] += 1
            K[w, fractional_offset + s + w - 1] += gp[s - 1]*gp[s]
    u, s, v = linalg.svd(K)
    trunc = 0
    v_trunc = v[:W-trunc, :]
    return v_trunc.T @ np.diag(1/s[:W-trunc]**2) @ v_trunc

n = 50
first = 0
for i in [4178]:#range(first, first + n):
    print(i)
    g = rand.default_rng(i).integers(0, 2, S)*2 - 1  # gamma
    gp = np.flip(g)  # gamma'
    c = calc(gp)
    c_res[i] = c

#%%

plt.axvline(fractional_offset, color="black", lw=1)
for i in [46, 4178]: # + list(range(95, 100)):
    plt.plot(np.diag(c_res[i]), label=f"seed {i}")
plt.xlim(0, (W + S - 2)*2 + 2)
plt.title("left: diagonal of covariance matrix for non-fractional lags,\nright: the same for first fractional")
plt.ylabel("diagonal value of $(X^TX)^{-1}$ (= covariance matrix)")
plt.xlabel("propotional to altitude of (fractional) lag value")
plt.legend()
plt.savefig("cov_by_seed.pdf")
plt.show()

#%%

for i in [46, 97]:
    g = rand.default_rng(i).integers(0, 2, S)*2 - 1
    plt.plot(np.correlate(g, g, "full"), label=f"seed {i}")
plt.legend()
plt.show()

#%%

c_approx_res = {}


#%%

def calc_approx(gp):
    K = np.zeros((W, S + W))
    for w in range(W):
        K[w, w] += 1
        K[w, w + 1] += 1
        for s in range(1, S):
            K[w, w + s] += 1 + 2*gp[s - 1]*gp[s]
            K[w, w + s + 1] += 1
    u, s, v = linalg.svd(K)
    trunc = 0
    v_trunc = v[:W-trunc, :]
    return v_trunc.T @ np.diag(1/s[:W-trunc]**2) @ v_trunc, K

n = 1
first = 0
for i in list(range(first, first + n)) + [4178]:
    print(i)
    g = rand.default_rng(i).integers(0, 2, S)*2 - 1  # gamma
    gp = np.flip(g)  # gamma'
    c_approx_res[i] = calc_approx(gp)

#%%

for i in [13, 46, 4178]:
    plt.plot(np.diag(c_approx_res[i][0]), label=f"seed {i}")
    #plt.plot(np.diag(c_res[i]), label=f"seed {i}", ls="--")
plt.xlim(0, S + W)
plt.title("diagonal of covariance matrix")
plt.ylabel("diagonal value of $(X^TX)^{-1}$ (= covariance matrix)")
plt.xlabel("propotional to altitude of (fractional) lag value")
plt.legend()
plt.savefig("cov_by_seed.pdf")
plt.show()

#%%

for i in [46, 97]:
    g = rand.default_rng(i).integers(0, 2, S)*2 - 1
    plt.plot(np.correlate(g, g, "full"), label=f"seed {i}")
plt.legend()
plt.show()

#%%

for i in [0, 1, 2, 46, 13]:
    c, K = c_approx_res[i]
    plt.plot(np.sqrt(np.sum((c @ K.T)**2, axis=1)), label=f"seed {i}")
plt.legend()
plt.show()

res = []
best_val, best_i = -1, -1
for i in range(10000):
    print(i, best_i, best_val)
    g = rand.default_rng(i).integers(0, 2, S)*2 - 1  # gamma
    gp = np.flip(g)  # gamma'
    r = np.trace(calc_approx(gp)[0])
    res.append((i, r))
    if best_i == -1 or r < best_val:
        best_i = i
        best_val = r
#%%

res2 = np.array(res)
s = np.argsort(res2[:, 1])
i = res2[s, 0]
v = res2[s, 1]




