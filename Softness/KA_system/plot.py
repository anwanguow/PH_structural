import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import linecache as lc

N = 864
N_frame = 1001;
gap = 10
T_min = 100
T_max = 900 + gap
sep = np.floor((T_max - T_min) / gap).astype(np.int16)
T = np.zeros(sep, dtype="float64")
T[0] = T_min
T[1] = T_min + gap
for i in range(2, sep):
    T[i] = T[i-1] + gap
T = T.astype(np.float32)
x = T

file_softness = "softness/softness.csv"
file_q6msd = "data/traj/nuc.log"

y_1 = np.zeros((sep), dtype="int")

y_2 = np.load("sep/sep.npy", allow_pickle=True)

y_3 = pd.read_csv(file_softness, header=None)
y_3 = np.transpose(y_3.values)

y_0 = np.zeros((sep), dtype="float32")

v_max_1 = np.zeros((N_frame), dtype="int")
v_max_2 = np.zeros((N_frame), dtype="float32")
for j in range(N_frame):
    txt = str.split(lc.getline(file_q6msd, 331 + j))
    num_str_1 = txt[6]
    if num_str_1 == "-1e+20":
        v_max_1[j] = 1
    else:
        v_max_1[j] = int(num_str_1)

    num_str_2 = txt[3]
    if num_str_2 == "-1e+20":
        v_max_2[j] = 0
    else:
        v_max_2[j] = np.float32(num_str_2)

for j in range(sep):
    idx = 100 + j * 10
    y_1[j] = v_max_1[idx]
    y_0[j] = v_max_2[idx]

plt.figure(figsize=[6, 6], dpi=300)
plt.xlabel("Time Step", fontsize=18)
# plt.ylabel("Quantities", fontsize=20)
plt.xticks([100,300,500,700,900],fontsize=18)
plt.yticks(fontsize=18)

plt.axhline(y=0, color='gray', linestyle='--')

# y_2 = (y_2 - np.min(y_2)) / (np.max(y_2) - np.min(y_2))

f0, = plt.plot(x, y_0 / np.max(y_0), label="NMSD($t$)", linestyle=':', linewidth=4)
f1, = plt.plot(x, y_1 / N, label="$Q_6(t)/\\mathscr{N}$", linestyle='-.', linewidth=4)
f2, = plt.plot(x, y_2, label="SI($t$)", linestyle='-', linewidth=4)
f3, = plt.plot(x, y_3, label="$\\mathcal{S}(t)$", linestyle='--', linewidth=4)
# plt.legend(handles=[f0, f1, f2, f3], fontsize=18, loc="best")

# plt.savefig("/home/chem/msrgxt/Desktop/ans.png", dpi=300)
plt.show()


