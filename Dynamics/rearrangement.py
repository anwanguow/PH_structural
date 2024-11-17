#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import MDAnalysis as md
from p_hop import p_hop

p_c = 0.1

t_R_2 = 20
t_R = 2 * t_R_2

# gap = 25
# T_min = 100
# T_max = 900+gap
# sep = np.floor((T_max - T_min) / gap).astype(np.int16)
# T = np.zeros(sep, dtype = "float64")
# T[0] = T_min
# T[1] = T_min + gap
# for i in range(2, sep):
#     T[i] = T[i-1] + gap
# T = T.astype(np.float32)
# t_set = T

t_set = np.array([100,200,300,400,500,600,700,800,900])

t_set = np.array([100, 200, 300, 400, 500, 600, 700, 800, 900])

for k in range(1, 3):
    for t_0 in t_set:
        T = np.zeros((t_R), dtype="int")
        for i in range(len(T)):
            T[i] = t_0 + i

        for num in range(10):
            source = "../data/Traj/Set_" + str(int(k)) + "/T_" + str(int(num)) + "/"
            traj_file = source + "traj_unwrap.dcd"
            u = md.lib.formats.libdcd.DCDFile(traj_file)
            N_frame = u.n_frames
            n_atom = u.header['natoms']
            traj = u.readframes()[0]

            hardsoft = np.zeros((len(traj[0])), dtype="int")

            p_list = np.zeros((len(traj[0]), len(T)), dtype="float32")

            for t in range(len(T)):
                col = p_list[:, t]
                col[:] = p_hop(traj, T[t], t_R_2)

            for i in range(len(hardsoft)):
                for j in range(len(T)):
                    if p_list[i][j] > p_c:
                        hardsoft[i] = 1
            
            save_dir = "p_hop_t/set_" + str(int(k)) + "/D_" + str(int(num)) + "/"
            
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            save_path = save_dir + "Y_" + str(int(t_0)) + ".npy"
            np.save(save_path, hardsoft, allow_pickle=True)
        
        print(k, "_", int(t_0))
