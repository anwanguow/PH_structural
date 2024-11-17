# -*- coding: utf-8 -*-

import numpy as np
import MDAnalysis as md
from p_hop import p_hop
import matplotlib.pyplot as plt

def compute_p_hop_values(index):
    num = index - 1
    source = "../data/Traj/set_1/T_" + str(int(num)) + "/"
    traj_file = source + "traj_unwrap.dcd"
    
    u = md.lib.formats.libdcd.DCDFile(traj_file)
    traj = u.readframes()[0]
    
    all_p_hop_values = []
    selected_frames = [100, 200, 300, 400, 500, 600, 700, 800, 900]
    
    for t in selected_frames:
        if t >= t_R_2 and t < len(traj) - t_R_2:
            p_hop_values = p_hop(traj, t, t_R_2)
            all_p_hop_values.extend(p_hop_values)
    
    return all_p_hop_values

t_R_2 = 20

# Compute hop probabilities for index 6 and 7
p_hop_values_index_6 = compute_p_hop_values(6)
p_hop_values_index_7 = compute_p_hop_values(7)

plt.figure(figsize=[8, 6], dpi=300)

weights_6 = np.ones_like(p_hop_values_index_6) / len(p_hop_values_index_6)
n_6, bins_6, patches_6 = plt.hist(p_hop_values_index_6, bins=40, range=(0, 0.6), weights=weights_6, color='#FF5733', edgecolor='black', alpha=0.7, label='$\\text{Traj}(T_6^{(1)})$')
weights_7 = np.ones_like(p_hop_values_index_7) / len(p_hop_values_index_7)
n_7, bins_7, patches_7 = plt.hist(p_hop_values_index_7, bins=40, range=(0, 0.6), weights=weights_7, color='#33FFCE', edgecolor='black', alpha=0.7, label='$\\text{Traj}(T_7^{(1)})$')

plt.axvline(x=0.1, color='#FF33A6', linestyle='--', linewidth=2)

plt.title("Histogram of $p_{hop}$ for $\\text{Traj}(T_6^{(1)})$ and $\\text{Traj}(T_7^{(1)})$", fontsize=18, color='black')
plt.xlabel("$p_{hop,i}(t)$", fontsize=18, color='black')
plt.ylabel("Frequency", fontsize=18, color='black')
plt.tick_params(axis='both', which='major', labelsize=14)
plt.xlim(bins_6[0], bins_6[-1])
plt.legend(loc='upper right', fontsize=16)

# plt.savefig("./p_hop.png", dpi=300)
plt.show()

