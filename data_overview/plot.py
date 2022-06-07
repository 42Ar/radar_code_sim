import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from scipy.io import loadmat

folders = glob('/data/code_zenith_1.00v_NO@vhf/20220506*')
folders.sort()
print(folders)
for folder in folders: 
    files = glob(folder + '/*')
    files.sort()
    date, power = [], []
    for f in files:
        print(f)
        m = loadmat(f)
        params = m['d_parbl'][0]
        power.append(params[7])
        date.append(int(f.split('/')[-1].split('.')[0]))
    plt.scatter(date, power, s=0.1)
    folder_name = folder.split('/')[-1]
    plt.savefig(f"overview_{folder_name}.pdf")
    plt.show()
