import numpy as np
import matplotlib.pyplot as plt
from config import N, n, c, cc, M, samples, rng
import signal_calc as sig

def doit(layer_gens):
    r = np.array([gen(rng) for gen in layer_gens])
    coded = r*np.tile(cc, N).reshape(r.shape)
    for i in range(1, N):
        coded[i, i:] = coded[i, :-i]
        coded[i, :i] = 0
    recieved = np.sum(coded, axis=0)
    res = []
    for h in range(N):
        L = len(c)
        filt = recieved[L + h:L + h + L]*c
        r = [np.vdot(filt[Delta:], filt[:-Delta])/L for Delta in range(1, M + 1)]
        res.append(r)
    return np.array(res)

t = np.arange(n)
acfs = [sig.gen_layer_ACF(t, i) for i in range(N)]
layer_gens = [sig.layer_generator(acf) for acf in acfs]
res = []
for s in range(samples):
    res.append(doit(layer_gens))
res = np.array(res)
mean = np.mean(res, axis=0)
std = np.std(res, axis=0)/np.sqrt(samples)

#%%

x = np.arange(1, M + 1)
t_plot = np.linspace(0, M)
for h, (m, s) in enumerate(zip(mean, std)):
    plt.plot(t_plot, sig.gen_layer_ACF(t_plot, h), color="black", lw=1)
    plt.errorbar(x, np.real(m), np.real(s), capsize=5,
                 ls="None", marker="x", label=f"{h}")
plt.xlim(0, M)
plt.ylim(-0.05, 1)
plt.xlabel("lag index")
plt.ylabel("power in a.u.")
plt.grid()
plt.legend()
plt.show()

#%%

from scipy.stats import shapiro

print(np.mean(res[:, 0, 0]))

ds = [np.real(res[:, 0, 0]), np.real(res[:, 1, 0])]
for d in ds:
    stat, p = shapiro(d)
    print(p)
    plt.hist(d, bins=50, alpha=0.5)
plt.show()
    
