import numpy as np
import matplotlib.pyplot as plt
from signal_calc import gen_layer_ACF, layer_generator
from config import N, n, cc, M, samples, w, off, c, code, max_fisher_size, mode, method, K
from scipy.stats import shapiro
from matplotlib.lines import Line2D
from analyze_barker_modulation.decoder import decode
from theo import fisher_matrix, theo_cov_with_cutoff
from scipy import linalg
from scipy.stats import pearsonr


def run_sim(layer_gens, rng):
    r = np.array([gen(rng) for gen in layer_gens])
    coded = r*np.tile(cc, N).reshape(r.shape)
    for i in range(1, N):
        coded[i, i:] = coded[i, :-i]
        coded[i, :i] = 0
    m = np.sum(coded, axis=0)
    res = decode(m[:, np.newaxis], off, w, c, N, M)
    return np.squeeze(res, axis=2)

rng = np.random.default_rng(1684658)
t = np.arange(n)
acfs = [gen_layer_ACF(t, i, N, mode) for i in range(N)]
layer_gens = [layer_generator(acf, n, method) for acf in acfs]
sim_res = []
for s in range(samples):  
    sim_res.append(run_sim(layer_gens, rng))
sim_res = np.array(sim_res)
mean = np.mean(np.real(sim_res), axis=0)
std = np.std(np.real(sim_res), axis=0)/np.sqrt(samples)
x = np.arange(1, M + 1)
t_plot = np.linspace(0, M)
for h, (m, s) in enumerate(zip(mean, std)):
    plt.plot(t_plot, gen_layer_ACF(t_plot, h, N, mode), color="black", lw=1)
    if code == "random":
        b = sum(acfs[k][1:M+1] for k in range(N) if k != h)/np.sqrt(w)
        plt.errorbar(x, acfs[h][1:M+1], b, ls="None", color="black", capsize=5)
    plt.errorbar(x, m, s, capsize=5, ls="None", marker="x", label=f"h={h}")
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
    F = np.diag(np.diag(fisher_matrix(c, acfs, N, M, w)))
    F_inv = linalg.inv(F)


#%%


colors = ["green", "brown", "blue", "orange", "grey", "darkblue"]
lower_bound = sum(acf[0] for acf in acfs)/np.sqrt(w)
lw=1
show_h = [0, 1]
for h, (m, s, color) in enumerate(zip(mean, std, colors)):
    if h not in show_h:
        continue
    plt.plot(x, s, marker="x", ls="-", color=color, lw=lw)
    theo_vars = np.array([theo_cov_with_cutoff(h, c, w, Delta, Delta, K, N, acfs) for Delta in range(1, M + 1)])
    theo_sig = np.sqrt(theo_vars/samples)
    plt.plot(x, theo_sig, ls="-.", color=color, marker="^")
    if len(c) <= max_fisher_size:
        fisher_limit = [np.sqrt(F_inv[h*M + Delta - 1, h*M + Delta - 1]/samples) for Delta in range(1, M + 1)]
        plt.plot(x, fisher_limit, marker="o", ls="--", color=color, lw=lw)
plt.xlabel("lag index")
plt.ylabel("standard deviation of power in a.u.")
legend_elements = [Line2D([0], [0], color=c, lw=lw, label=f'h={h}') for h, c in enumerate(colors) if h in show_h and h in range(N)]
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


r_table = np.zeros((M, M))
theo_table = np.zeros((M, M))
h = 0
theo_vars = np.array([theo_cov_with_cutoff(h, c, w, Delta, Delta, K, N, acfs) for Delta in range(1, M + 1)])

for i in range(M - 1):
    for j in range(i + 1, M):
        r, p = pearsonr(np.real(sim_res[:, h, i]), np.real(sim_res[:, h, j]))
        r_table[i, j] = r
        r_table[j, i] = r
        r_theo = theo_cov_with_cutoff(h, c, w, i + 1, j + 1, K, N, acfs)/np.sqrt(theo_vars[i]*theo_vars[j])
        theo_table[i, j] = r_theo
        theo_table[j, i] = r_theo
for i in range(M):
    r_table[i, i] = np.nan
    theo_table[i, i] = np.nan

fig, axs = plt.subplots(1, 3)
for ax, t in zip(axs, (r_table, theo_table)):
    img = ax.imshow(t, extent=(0.5, M + 0.5, 0.5, M + 0.5), vmin=0, vmax=1)
    plt.colorbar(img, ax=ax)
img = axs[2].imshow(theo_table - r_table)
plt.colorbar(img, ax=axs[2])
axs[0].set_title("correlations from simulation")
axs[1].set_title("theoretical correlations")
axs[2].set_title("theoretical - simulation")
for ax in axs:
    ax.set_xlabel("lag index")
    ax.set_ylabel("lag index")
plt.show()



#%%

check_for_gaussian_dist = True
if check_for_gaussian_dist:
    ds = [np.real(sim_res[:, 1, 2]), np.real(sim_res[:, 1, 5])]
    for d in ds:
        stat, p = shapiro(d)
        print(p)
        plt.hist(d, bins=50, alpha=0.5)
    plt.show()
