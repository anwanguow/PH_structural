# -*- coding: utf-8 -*-

import numpy as np
import MDAnalysis as md
from p_hop import p_hop
import matplotlib.pyplot as plt

t_R_2 = 20
t_R = 2 * t_R_2

index = 1
num = index - 1

source = "../data/Traj/set_1/T_" + str(int(num)) + "/"
traj_file = source + "traj_unwrap.dcd"
u = md.lib.formats.libdcd.DCDFile(traj_file)
N_frame = u.n_frames
n_atom = u.header['natoms']
traj = u.readframes()[0]

p_list = np.zeros((len(traj[0]), len(traj)), dtype="float32")

for t in range(t_R_2, len(traj) - t_R_2):
    col = p_list[:, t]
    col[:] = p_hop(traj, t, t_R_2)

threshold = 0.1

rearrangements = []

for particle_index in range(len(traj[0])):
    particle_data = p_list[particle_index]
    x0 = None

    for t in range(len(particle_data)):
        if particle_data[t] < threshold:
            if x0 is not None:
                rearrangements.append(t - x0)
                x0 = None
        elif x0 is None and particle_data[t] >= threshold:
            x0 = t

plt.figure(figsize=[10, 7], dpi=300)
bins = np.linspace(0, max(rearrangements), 31)
n, bins, patches = plt.hist(rearrangements, bins=bins, edgecolor='black', color='#FF5733', alpha=0.7, density=True)
plt.axvline(x=40, color='#FF33A6', linestyle='--', linewidth=2)

plt.title("Histogram of Rearrangement Durations in $\\text{Traj}(T^{(1)}_1)$", fontsize=22, color='black')
plt.xlabel("Rearrangement Duration", fontsize=20, color='black')
plt.ylabel("Frequency", fontsize=20, color='black')

plt.tick_params(axis='both', which='major', labelsize=16)

# plt.savefig("./durations.png", dpi=300)
plt.show()