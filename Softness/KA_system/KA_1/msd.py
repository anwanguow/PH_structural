#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import linecache as lc

N_frame = 1001;

N = np.zeros((N_frame), dtype = "int")
for i in range(N_frame):
    N[i] = i + 1
    
num = 1

parameter = np.zeros((num), dtype = "object")
f = np.zeros((num), dtype = "object")

plt.figure(figsize=[6,8])

for i in range(num):
    name = "data/traj/nuc.log"
    v_max_n = np.zeros((N_frame), dtype="float32")
    for j in range(N_frame):
        txt = str.split(lc.getline(name, 331 + j))
        num_str = txt[3];
        if num_str == "8.2173011e-31":
            v_max_n[j] = 1
        else:
            v_max_n[j] = float(num_str)
    parameter[i] = v_max_n
    f[i] ,= plt.plot(N, parameter[i])


plt.xlabel("Time $100*(0.2*(m*\sigma^2/\epsilon)^{1/2})$")
plt.ylabel("MSD")

plt.xlim([100,900])


plt.savefig("/home/chem/msrgxt/Desktop/msd.png", dpi = 300)
plt.show()



