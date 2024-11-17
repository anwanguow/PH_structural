#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import linecache as lc

N_frame = 1001;

N = np.zeros((N_frame), dtype = "int")
for i in range(N_frame):
    N[i] = i + 1
    
num = 10

parameter = np.zeros((num), dtype = "object")
f = np.zeros((num), dtype = "object")

plt.figure(figsize=[4,8])

for i in range(num):
    name = "T_" + str(i) + "/nuc.log"
    v_max_n = np.zeros((N_frame), dtype="float64")
    for j in range(N_frame):
        txt = str.split(lc.getline(name, 311 + j))
        num_str = txt[3];
        if num_str == "-1e+20":
            v_max_n[j] = 0
        else:
            v_max_n[j] = np.float64(num_str)
    parameter[i] = v_max_n
    f[i] ,= plt.plot(N, parameter[i])

plt.xlim([100,900])
plt.title("Mean Square Displacement (MSD)")
plt.xlabel("Time $100*(0.2*(m*\sigma^2/\epsilon)^{1/2})$")
plt.ylabel("Mean Squared Displacement (MSD)")
plt.legend(handles=[f[0],f[1],f[2],f[3],f[4],f[5],f[6],f[7],f[8],f[9]], labels = ["Traj$(T^{(2)}_{1})$","Traj$(T^{(2)}_{2})$","Traj$(T^{(2)}_{3})$","Traj$(T^{(2)}_{4})$","Traj$(T^{(2)}_{5})$","Traj$(T^{(2)}_{6})$","Traj$(T^{(2)}_{7})$","Traj$(T^{(2)}_{8})$","Traj$(T^{(2)}_{9})$","Traj$(T^{(2)}_{10})$"], loc="best")
plt.savefig("/home/chem/msrgxt/Desktop/msd.png", dpi = 300)
plt.show()


