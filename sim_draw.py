# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

N = 2
len_g = 2
S = 3
layer_cnt = N*(len_g + S - 1)
tx_number = 2

for i in range(tx_number*(len_g + S - 1)):
    for j in range(tx_number*(len_g + S - 1)):
        color = "orange" if i%(len_g + S - 1) < len_g else "lightblue"
        plt.gca().add_patch(Rectangle((i + 0.5*j, 0.5*j - 0.5), 1/np.sqrt(2), 1/np.sqrt(2), angle=45, edgecolor="black", facecolor=color))


plt.xlim(-0.6, 11)
plt.ylim(-0.6, 5)

for i in range(len_g):
    plt.annotate(f"$\\gamma_{i}$", (-0.6 + i, -0.5))
for i in range(S):
    plt.annotate(f"$m_{i}$", (-0.72 + i + len_g, -0.5))
for n in range(N):
    for i in range(S):
        for j in range(len_g):
            h = n*(len_g + S - 1) + i + j
            plt.annotate(f"$V^{{({n}, {h})}}_{{0, {j}}}$", (len_g - 1.4 + h*0.5 - j, -0.3 + 0.5*h))


plt.axis('off')
plt.gca().set_aspect('equal')


