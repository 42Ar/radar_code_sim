# -*- coding: utf-8 -*-

from glob import glob
from pathlib import Path
import matplotlib.pyplot as plt

files = glob("/data/barker_pulses/final_original_code/10*")
number = [int(Path(f).stem) for f in files]
number.sort()
plt.plot(number, ls="None", marker="x")
plt.show()

print(" ".join(f"{n}.mat" for n in number[549:]))
