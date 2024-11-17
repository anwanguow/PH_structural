# -*- coding: utf-8 -*-

import numpy as np
import MDAnalysis as md
from p_hop import p_hop
import matplotlib.pyplot as plt


t_R_2 = 20
t_R = 2 * t_R_2


index = 1
num = index-1
particle_index = 310

source = "../data/Traj/set_1/T_" + str(int(num)) + "/"
traj_file = source + "traj_unwrap.dcd"
u = md.lib.formats.libdcd.DCDFile(traj_file)
N_frame = u.n_frames
n_atom = u.header['natoms']
traj = u.readframes()[0]

p_list = np.zeros((len(traj[0]), len(traj)), dtype = "float32")

for t in range(t_R_2, len(traj)-t_R_2):
    col = p_list[:,t]
    col[:] = p_hop(traj, t, t_R_2)

plt.figure(figsize=[10,7], dpi=300)
plt.plot(p_list[particle_index])
plt.xlim([150,240])
plt.axhline(y=0.1, color='r', linestyle='--')
plt.axvspan(170, 210, color='yellow', alpha=0.2)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel("Time $100*(0.2*(m*\sigma^2/\epsilon)^{1/2})$", fontsize=24)
plt.ylabel("$p_{hop,i}(t)$", fontsize=24)
plt.title("The values of $p_{hop,310}(t)$ in $\\text{Traj}(T^{(1)}_1)$", fontsize=24)
# plt.savefig("./p_hop_t.png", dpi=300)
plt.show()
