import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

loops = 25
c = [None]*48  # only the length matters here
g = [None]*66  # only the length matters here
calib_samples = 10
oversample_factor = 2
samples_per_subcycle = 1216
samples_after_RFOFF = 10

#infiles = ["10667616.mat", "10671098.mat", "10671286.mat", "10690765.mat", "10692248.mat", "10733552.mat2"]
#infiles = ["10733552.mat"]  # newest
#infiles = ["10671286.mat"]  # modulated
infiles = ["/home/frank/study/radar_code/radar_code_sim/data_ana_test_exp/10742247.mat"]  # modulated
infiles = ["/home/frank/study/radar_code/radar_code_sim/data_ana_test_exp/10744489.mat"]  # modulated
infiles = ["/home/frank/study/radar_code/radar_code_sim/data_ana_test_exp/10745445.mat"]  # modulated
infiles = ["/home/frank/study/radar_code/radar_code_sim/data_ana_test_exp/10745929.mat"]  # modulated
infiles = ["/home/frank/study/radar_code/radar_code_sim/data_ana_test_exp/10749834.mat"]  # modulated
infiles = ["/home/frank/study/radar_code/radar_code_sim/data_ana_test_exp/10750194.mat"]  # modulated


file_data = []
P = []
for i, file in enumerate(infiles):
    print(f"reading {i}:", file)
    file = loadmat(file)
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
    m = file['d_raw'][:, 0]
    assert (len(g)*oversample_factor + calib_samples + samples_after_RFOFF + samples_per_subcycle)*len(c)*loops*2 == len(m)
    file_data.append(m)

#%%

calib_len = len(g)*oversample_factor + samples_after_RFOFF + calib_samples
total_calib_count = calib_len*len(c)*loops
total_sample_count = samples_per_subcycle*len(c)*loops
n = 400
for antenna_off in [0, total_calib_count + total_sample_count]:
    plt.plot(np.real(file_data[0][antenna_off:antenna_off+n]))
    for i in range(2):
        plt.axvline(calib_len*i + len(g)*oversample_factor + samples_after_RFOFF, c="red")
        plt.axvline(calib_len*i + len(g)*oversample_factor + samples_after_RFOFF + calib_samples, c="green")
plt.title("calibration data: comparison of both antenna halfs")
plt.show()

#%%

calib = file_data[0][:total_calib_count].reshape(len(c)*loops, calib_len)
calib = np.mean(np.abs(calib[:, len(g)*oversample_factor + samples_after_RFOFF:]), axis=1)
plt.scatter(np.arange(len(c)*loops), calib)
plt.show()

#%%

n = 1000
plt.scatter(np.arange(n), np.real(file_data[0][total_calib_count:total_calib_count + n]), s=0.5, label="antenna 1")
plt.scatter(np.arange(n), np.real(file_data[0][total_sample_count + 2*total_calib_count:total_sample_count + 2*total_calib_count + n]), s=0.5, label="antenna 2")
plt.legend()
plt.title("zoom into data region")
plt.show()
