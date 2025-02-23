import numpy as np
import matplotlib.pyplot as plt
import linecache as lc

N = 864
N_frame = 101
gap = 1
T_min = 0
T_max = 100 + gap
sep = np.floor((T_max - T_min) / gap).astype(np.int16)
T = np.zeros(sep, dtype="float64")
T[0] = T_min
T[1] = T_min + gap
for i in range(2, sep):
    T[i] = T[i-1] + gap
T = T.astype(np.float32)
x = T

file_q6msd = "data/traj/nuc.log"
y_1 = np.zeros((sep), dtype="int")
full_y_2 = np.load("results/SI.npy", allow_pickle=True)


indices = ((x - T_min) / gap).astype(int)
y_2 = full_y_2[indices]

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

y_1 = v_max_1
y_0 = v_max_2

plt.figure(figsize=[8, 7], dpi=300)
plt.xlabel("Time Step", fontsize=18)
plt.xticks([0, 20, 40, 60, 80, 100], fontsize=18)
plt.yticks(fontsize=18)


y_0 = (y_0-np.min(y_0))/(np.max(y_0)-np.min(y_0))

plt.axhline(y=0, color='gray', linestyle='--')

f1, = plt.plot(x, y_1/N, label="$Q_6(t)/\\mathscr{N}$", linestyle='-.', linewidth=3)
f2, = plt.plot(x, y_2, label="SI($t$)", linestyle='dashdot', linewidth=3)
f3, = plt.plot(x, y_0, label="NMSD", linestyle='-', linewidth=3)
plt.legend(fontsize=16, loc="best")
plt.savefig("Figure.png",bbox_inches='tight')
plt.show()

