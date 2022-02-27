import numpy as np
import math
from scipy import linalg
import matplotlib.pyplot as plt

# M = 5, N = 7
c = [-1, +1, -1, -1, +1, +1, +1, +1, +1, +1, +1, +1, +1, -1, +1, +1, -1, -1, +1, -1, +1, +1, -1, -1, -1, +1, +1, -1, +1, +1, +1, -1, +1, -1, -1, -1, -1, +1, +1, +1]
n = len(c)*3
N = 7
M = 8
cc = np.tile(c, math.ceil(n/len(c)))[:n]
rng = np.random.default_rng(24)
samples = 1000

#        #   #   #
#       / \ / \ / \
#      #   #   #   \
#     / \ / \ / \   \
#    #   #   #   \   \
#   / \ / \ / \   \   \

    
def gen_layer_ACF(t, i):
    width = 3*i/N + 1
    return (i/N + 1)*np.exp(-t/width)

def gen_layer_matrix(ACF):
    M = linalg.toeplitz(ACF)
    T = linalg.sqrtm(M)
    return T

def gen_layer(T):
    a = np.matmul(T, rng.normal(size=n))
    b = np.matmul(T, rng.normal(size=n))
    return a + 1j*b

def doit(layer_matrices):
    r = np.array([gen_layer(T) for T in layer_matrices])
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
acfs = [gen_layer_ACF(t, i) for i in range(N)]
layers = [gen_layer_matrix(acf) for acf in acfs]
res = []
for s in range(samples):
    res.append(doit(layers))
res = np.array(res)
mean = np.mean(res, axis=0)
std = np.std(res, axis=0)/np.sqrt(samples)

#%%

x = np.arange(1, M + 1)
t_plot = np.linspace(0, M)
for h, (m, s) in enumerate(zip(mean, std)):
    plt.plot(t_plot, 2*gen_layer_ACF(t_plot, h), color="black", lw=1)
    plt.errorbar(x, np.real(m), np.real(s), capsize=5,
                 ls="None", marker="x", label=f"{h}")
plt.xlim(0, M)
plt.legend()
    
    
