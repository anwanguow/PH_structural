#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import linecache as lc

set_choose = 2
num = 10

N_frame = 1001
N = np.zeros((N_frame), dtype="int")
for i in range(N_frame):
    N[i] = i + 1

parameter = np.zeros((num), dtype="object")
f = []

plt.figure(figsize=[10, 8], dpi=300)

for i in range(num):
    name = "data/Traj/Set_" + str(set_choose) + "/T_" + str(i) + "/nuc.log"
    v_max_n = np.zeros((N_frame), dtype="int")
    for j in range(N_frame):
        txt = str.split(lc.getline(name, 311 + j))
        num_str = txt[6]
        if num_str == "-1e+20":
            v_max_n[j] = 1
        else:
            v_max_n[j] = int(num_str)
    parameter[i] = v_max_n
    line, = plt.plot(N, parameter[i])
    f.append(line)

plt.xlabel("Time $100*(0.2*(m*\sigma^2/\epsilon)^{1/2})$")
plt.ylabel("Global bond order parameter $Q_6$")

plt.xlim([100, 900])

legend_labels = [f"Traj$(T^{{({set_choose})}}_{{{i+1}}})$" for i in range(num)]
plt.legend(handles=f, labels=legend_labels, loc="best")

plt.show()
