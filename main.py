import numpy as np
import matplotlib.pyplot as plt
from config import N, n, cc, M, samples, w, off, c
import signal_calc as sig
from scipy import linalg

def doit(layer_gens, rng):
    r = np.array([gen(rng) for gen in layer_gens])
    coded = r*np.tile(cc, N).reshape(r.shape)
    for i in range(1, N):
        coded[i, i:] = coded[i, :-i]
        coded[i, :i] = 0
    recieved = np.sum(coded, axis=0)
    res = []
    for h in range(N):
        d1 = recieved[off + h:off + h + w]*cc[off:off + w]
        r = []
        for Delta in range(1, M + 1):
            d2 = recieved[off + h + Delta:off + h + Delta + w]*cc[off + Delta:off + Delta + w]
            r.append(np.vdot(d2, d1)/w)
        res.append(r)
    return np.array(res)

rng = np.random.default_rng(2)
t = np.arange(n)
acfs = [sig.gen_layer_ACF(t, i) for i in range(N)]
layer_gens = [sig.layer_generator(acf) for acf in acfs]
res = []
for s in range(samples):  
    res.append(doit(layer_gens, rng))
res = np.array(res)
mean = np.mean(res, axis=0)
std = np.std(res, axis=0)/np.sqrt(samples)
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
plt.savefig("code_sim_res.pdf")
plt.show()

#%%

sigma = np.sum([np.outer(np.roll(cc, k), np.roll(cc, k))*linalg.toeplitz(acf)
               for k, acf in enumerate(acfs)], axis=0)
sigma_inv = linalg.inv(sigma)


#%%

def sigma_deriv(k_p, Delta_p):
    i = np.repeat(np.arange(n), n).reshape(n, n)
    j = np.tile(np.arange(n), n).reshape(n, n)
    delta = np.abs(i-j) == Delta_p
    return np.outer(np.roll(cc, k_p), np.roll(cc, k_p))*delta

def fisher(k1, k2, Delta1, Delta2):
    A = np.matmul(sigma_inv, sigma_deriv(k1, Delta1))
    B = np.matmul(sigma_inv, sigma_deriv(k2, Delta2))
    return np.trace(np.matmul(A, B))/2

def full_fisher():
    r = np.zeros((N*M, N*M))
    for k1 in range(N):
        for Delta1 in range(1, M+1):
            for k2 in range(N):
                for Delta2 in range(1, M+1):
                    r[k1*M + Delta1 - 1, k2*M + Delta2 - 1] = fisher(k1, k2, Delta1, Delta2)
    return r


F = full_fisher()
F_inv = linalg.inv(F)

#%%

sigma2 = np.sum([np.outer(np.roll(np.tile(c, 3), k), np.roll(np.tile(c, 3), k))*linalg.toeplitz(acf[:3*len(c)])
                 for k, acf in enumerate(acfs)], axis=0)
sigma2_inv = linalg.inv(sigma2)

def fisher2(k_i, k_j, Delta_i, Delta_j):
    eps_i = np.roll(c, k_i)*np.roll(c, k_i - Delta_i)
    eps_j = np.roll(c, k_j)*np.roll(c, k_j - Delta_j)
    eps_LL = np.outer(eps_i, eps_j)
    # square matrix of size len(c), containg diagonal entries
    a = np.sum(eps_LL*sigma2_inv[len(c):2*len(c), len(c):2*len(c)]*
               sigma2_inv[len(c)+Delta_i:2*len(c)+Delta_i, len(c)+Delta_j:2*len(c)+Delta_j])
    # also square matrix of size len(c), located directly next to the first matrix
    b = np.sum(eps_LL[:,:-M]*sigma2_inv[len(c):2*len(c), 2*len(c):3*len(c)-M]*
               sigma2_inv[len(c)+Delta_i:2*len(c)+Delta_i, 2*len(c)+Delta_j:3*len(c)+Delta_j - M])
    return (a + 2*b)*n/len(c)/2

def full_fisher2():
    F = np.zeros((N*M, N*M))
    for k1 in range(N):
        for Delta1 in range(1, M+1):
            for k2 in range(N):
                for Delta2 in range(1, M+1):
                    p1 = k1*M + Delta1 - 1
                    p2 = k2*M + Delta2 - 1
                    if p2 >= p1:
                        F[p1, p2] = (fisher2(k1, k2, Delta1, Delta2) +
                                     fisher2(k1, k2, Delta1, -Delta2) +
                                     fisher2(k1, k2, -Delta1, Delta2) +
                                     fisher2(k1, k2, -Delta1, -Delta2))
                    else:
                        F[p1, p2] = F[p2, p1]
    return F

F2 = full_fisher2()
F2_inv = linalg.inv(F2)

rng = np.random.default_rng(456318)

theo_bound = sum(acf[0] for acf in acfs)/np.sqrt(2*len(cc))
theo_bound_est = sum(acf[0] for acf in acfs)/np.sqrt(len(cc))

for h, (m, s) in enumerate(zip(mean, std)):
    #plt.plot(x, np.real(s), marker="x", ls="-", label=f"est. h={h}")
    theo_limit = [np.sqrt(F_inv[h*M + Delta - 1, h*M + Delta - 1]/samples) for Delta in range(1, M + 1)]
    plt.plot(x, theo_limit, marker="o", ls="--", label=f"theo. h={h}")
    theo_limit = [np.sqrt(F2_inv[h*M + Delta - 1, h*M + Delta - 1]/samples) for Delta in range(1, M + 1)]
    plt.plot(x, theo_limit, marker="x", ls="-", label=f"theo2. h={h}")
plt.xlabel("lag index")
plt.ylabel("standard deviation of power in a.u.")
plt.legend()
plt.axhline(theo_bound/np.sqrt(samples), label="cramer rao bound")
plt.axhline(theo_bound_est/np.sqrt(samples), label="perefect code")
plt.savefig("code_normal_case.pdf")
plt.show()


#%%

from scipy.stats import shapiro

ds = [np.real(res[:, 0, 0]), np.real(res[:, 1, 0])]
for d in ds:
    stat, p = shapiro(d)
    print(p)
    plt.hist(d, bins=50, alpha=0.5)
plt.show()
    
