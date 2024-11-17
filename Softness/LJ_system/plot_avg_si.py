#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

N = 864
N_frame = 1001
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

data_group_1 = []
data_group_2 = []

plt.figure(figsize=[6, 8], dpi=300)
plt.xlabel("Time Step", fontsize=18)
plt.ylabel("Separation Index (SI)", fontsize=18)
plt.xticks([100, 300, 500, 700, 900], fontsize=18)
plt.yticks(fontsize=18)
plt.axhline(y=0, color='gray', linestyle='--')

for num in range(1, 11):
    y_2 = np.load("sep/sep_" + str(num - 1) + ".npy", allow_pickle=True)
    if num <= 6:
        data_group_1.append(y_2)
    else:
        data_group_2.append(y_2)
    plt.plot(x, y_2, label=fr"$\text{{Traj}}(T^{{(2)}}_{{{num}}})$", linestyle='--', linewidth=1, alpha=0.3)

mean_group_1 = np.mean(data_group_1, axis=0)
mean_group_2 = np.mean(data_group_2, axis=0)

plt.plot(x, mean_group_1, linestyle='-', linewidth=3)
plt.plot(x, mean_group_2, linestyle='-', linewidth=3)

plt.legend(fontsize=14, loc="best")
plt.show()
