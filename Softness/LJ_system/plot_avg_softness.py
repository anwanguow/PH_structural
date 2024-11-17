#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
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
plt.ylabel(r"Global Softness ($\mathcal{S}(t)$)", fontsize=18)
plt.xticks([100, 300, 500, 700, 900], fontsize=18)
plt.yticks(fontsize=18)
plt.axhline(y=0, color='gray', linestyle='--')

for num in range(1, 11):
    file_softness = f"softness_2/softness_t_{num-1}.csv"
    y_3 = pd.read_csv(file_softness, header=None).values.flatten()

    if num <= 6:
        data_group_1.append(y_3)
    else:
        data_group_2.append(y_3)

    plt.plot(x, y_3, linestyle='--', linewidth=1, alpha=0.3)

mean_group_1 = np.mean(data_group_1, axis=0)
mean_group_2 = np.mean(data_group_2, axis=0)

line1, = plt.plot(x, mean_group_1, linestyle='-', linewidth=3, label="Category I")
line2, = plt.plot(x, mean_group_2, linestyle='-', linewidth=3, label="Category II")

plt.legend(handles=[line1, line2], fontsize=18, loc="lower left")

plt.show()
