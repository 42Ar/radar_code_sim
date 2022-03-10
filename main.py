import numpy as np
import matplotlib.pyplot as plt
from config import N, n, cc, M, samples, w, off, c
import signal_calc as sig
from scipy import linalg
from scipy.stats import shapiro
from matplotlib.lines import Line2D


def run_sim(layer_gens, rng):
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
sim_res = []
for s in range(samples):  
    sim_res.append(run_sim(layer_gens, rng))
sim_res = np.array(sim_res)
mean = np.mean(sim_res, axis=0)
std = np.std(sim_res, axis=0)/np.sqrt(samples)
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

sigma2 = np.sum([np.outer(np.roll(np.tile(c, 3), k), np.roll(np.tile(c, 3), k))*linalg.toeplitz(acf[:3*len(c)])
                 for k, acf in enumerate(acfs)], axis=0)
sigma2_inv = linalg.inv(sigma2)

def fisher(k_i, k_j, Delta_i, Delta_j):
    eps_i = np.roll(c, k_i)*np.roll(c, k_i - Delta_i)
    eps_j = np.roll(c, k_j)*np.roll(c, k_j - Delta_j)
    eps_LL = np.outer(eps_i, eps_j)
    # square matrix of size len(c), containg diagonal entries
    a = np.sum(eps_LL*sigma2_inv[len(c):2*len(c), len(c):2*len(c)]*
               sigma2_inv[len(c)+Delta_i:2*len(c)+Delta_i, len(c)+Delta_j:2*len(c)+Delta_j])
    # also square matrix of size len(c), located directly next to the first matrix
    b = np.sum(eps_LL[:,:-M]*sigma2_inv[len(c):2*len(c), 2*len(c):3*len(c)-M]*
               sigma2_inv[len(c)+Delta_i:2*len(c)+Delta_i, 2*len(c)+Delta_j:3*len(c)+Delta_j - M])
    return (a + 2*b)*w/len(c)/2

def fisher_matrix():
    F = np.zeros((N*M, N*M))
    for k1 in range(N):
        for Delta1 in range(1, M+1):
            for k2 in range(N):
                for Delta2 in range(1, M+1):
                    p1 = k1*M + Delta1 - 1
                    p2 = k2*M + Delta2 - 1
                    if p2 >= p1:
                        F[p1, p2] = (fisher(k1, k2, Delta1, Delta2) +
                                     fisher(k1, k2, Delta1, -Delta2) +
                                     fisher(k1, k2, -Delta1, Delta2) +
                                     fisher(k1, k2, -Delta1, -Delta2))
                    else:
                        F[p1, p2] = F[p2, p1]
    return F

F = fisher_matrix()
F_inv = linalg.inv(F)

#%%

colors = ["green", "brown", "blue", "orange", "grey", "darkblue"]
lower_bound = sum(acf[0] for acf in acfs)/np.sqrt(w)
lw=1.4
for h, (m, s, color) in enumerate(zip(mean, std, colors)):
    plt.plot(x, np.real(s), marker="x", ls="-", color=color, lw=lw)
    fisher_limit = [np.sqrt(F_inv[h*M + Delta - 1, h*M + Delta - 1]/samples) for Delta in range(1, M + 1)]
    plt.plot(x, fisher_limit, marker="o", ls="--", color=color, lw=lw)
plt.xlabel("lag index")
plt.ylabel("standard deviation of power in a.u.")
limit = plt.axhline(lower_bound/np.sqrt(samples), lw=lw, label="infinite\nrandom\ncode", color="red")
legend_elements = [Line2D([0], [0], color=c, lw=lw, label=f'h={h}') for h, c in enumerate(colors)]
legend_elements.append(limit)
legend_elements.append(Line2D([0], [0], color=colors[0], lw=lw, ls="-", marker="x", label='Monte-Carlo'))
legend_elements.append(Line2D([0], [0], color=colors[0], lw=lw, ls="--", marker="o", label='Cramér–Rao'))
plt.gca().legend(handles=legend_elements, loc='right')
plt.xticks(range(1, M+1), [f" {i}" for i in range(1, M+1)])
plt.savefig("code_normal_case.pdf")
plt.show()


#%%


ds = [np.real(sim_res[:, 0, 0]), np.real(sim_res[:, 1, 0])]
for d in ds:
    stat, p = shapiro(d)
    print(p)
    plt.hist(d, bins=50, alpha=0.5)
plt.show()
    
