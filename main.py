import numpy as np
import matplotlib.pyplot as plt
from config import N, n, cc, M, samples, w, off, c, code, max_fisher_size
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
    if code == "random":
        b = sum(acfs[k][1:M+1] for k in range(N) if k != h)/np.sqrt(w)
        plt.errorbar(x, acfs[h][1:M+1], b, ls="None", color="black", capsize=5)
    plt.errorbar(x, np.real(m), np.real(s), capsize=5,
                 ls="None", marker="x", label=f"h={h}")
plt.xlim(0, M)
plt.ylim(-0.05, 1)
plt.xlabel("lag index")
plt.ylabel("power in a.u.")
plt.grid()
plt.legend()
plt.title(f"samples={samples}, N={N}, M={M}, L={len(c)}, repetitions={w/len(c)}")
plt.savefig("code_sim_res.pdf", bbox_inches='tight')
plt.show()


#%%

if len(c) <= max_fisher_size:
    def pad_acf(acf, length):
        if len(acf) >= length:
            return acf[:length]
        r = np.zeros(length)
        r[:len(acf)] = acf
        return r

    sigma = np.sum([np.outer(np.roll(np.tile(c, 3), k), np.roll(np.tile(c, 3), k))*linalg.toeplitz(pad_acf(acf, 3*len(c)))
                     for k, acf in enumerate(acfs)], axis=0)
    sigma_inv = linalg.inv(sigma)

    def fisher(k_i, k_j, Delta_i, Delta_j):
        eps_i = np.roll(c, k_i)*np.roll(c, k_i - Delta_i)
        eps_j = np.roll(c, k_j)*np.roll(c, k_j - Delta_j)
        eps_LL = np.outer(eps_i, eps_j)
        # square matrix of size len(c), containg diagonal entries
        a = np.sum(eps_LL*sigma_inv[len(c):2*len(c), len(c):2*len(c)]*
                   sigma_inv[len(c)+Delta_i:2*len(c)+Delta_i, len(c)+Delta_j:2*len(c)+Delta_j])
        # also square matrix of size len(c), located directly next to the first matrix
        b = np.sum(eps_LL[:,:-M]*sigma_inv[len(c):2*len(c), 2*len(c):3*len(c)-M]*
                   sigma_inv[len(c)+Delta_i:2*len(c)+Delta_i, 2*len(c)+Delta_j:3*len(c)+Delta_j - M])
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

    F = np.diag(np.diag(fisher_matrix()))
    F_inv = linalg.inv(F)


#%%

def theo_var(h, Delta):
    var = 0
    for k1 in range(N):
        for k2 in range(N):
            eps_hat = cc[off                 :off + w]*\
                      cc[off - Delta         :off + w - Delta]*\
                      cc[off - h + k2        :off + w - h + k2]*\
                      cc[off - Delta - h + k1:off + w - Delta - h + k1]
            var += np.sum(np.outer(eps_hat, eps_hat)*
                          linalg.toeplitz(acfs[k1][:w])*
                          linalg.toeplitz(acfs[k2][:w]))
    return var/w**2

colors = ["green", "brown", "blue", "orange", "grey", "darkblue"]
lower_bound = sum(acf[0] for acf in acfs)/np.sqrt(w)
lw=1
show_h = [0, 5]
for h, (m, s, color) in enumerate(zip(mean, std, colors)):
    if h not in show_h:
        continue
    plt.plot(x, np.real(s), marker="x", ls="-", color=color, lw=lw)
    theo_sig = [np.sqrt(theo_var(h, Delta)/samples) for Delta in range(1, M + 1)]
    plt.plot(x, theo_sig, ls="-.", color=color, marker="^")
    if len(c) <= max_fisher_size:
        fisher_limit = [np.sqrt(F_inv[h*M + Delta - 1, h*M + Delta - 1]/samples) for Delta in range(1, M + 1)]
        plt.plot(x, fisher_limit, marker="o", ls="--", color=color, lw=lw)
plt.xlabel("lag index")
plt.ylabel("standard deviation of power in a.u.")
legend_elements = [Line2D([0], [0], color=c, lw=lw, label=f'h={h}') for h, c in enumerate(colors) if h in show_h]
limit = plt.axhline(lower_bound/np.sqrt(samples), lw=lw, label="infinite\nrandom\ncode", color="grey")
legend_elements.append(limit)
legend_elements.append(Line2D([0], [0], color=colors[show_h[0]], lw=lw, ls="-", marker="x", label='Monte-Carlo'))
legend_elements.append(Line2D([0], [0], color=colors[show_h[0]], lw=lw, ls="--", marker="o", label='Cramér–Rao'))
legend_elements.append(Line2D([0], [0], color=colors[show_h[0]], lw=lw, ls="-.", marker="^", label='Analytical'))
plt.gca().legend(handles=legend_elements, loc='right')
plt.xticks(range(1, M+1), [f" {i}" for i in range(1, M+1)])
plt.savefig("code_normal_case.pdf", bbox_inches='tight')
plt.show()


#%%

check_for_gaussian_dist = False
if check_for_gaussian_dist:
    ds = [np.real(sim_res[:, 0, 0]), np.real(sim_res[:, 1, 0])]
    for d in ds:
        stat, p = shapiro(d)
        print(p)
        plt.hist(d, bins=50, alpha=0.5)
    plt.show()
