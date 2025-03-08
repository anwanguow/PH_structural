# -*- coding: utf-8 -*-
import os
import numpy as np
import MDAnalysis as md
from scipy.spatial import cKDTree
from scipy.special import sph_harm

threshold_q6 = 0.4
t_set = np.array([100, 200, 300, 400, 500, 600, 700, 800, 900])

def compute_q6(universe, frame, cutoff=1.3):
    universe.trajectory[frame]
    positions = universe.atoms.positions[:, :3]
    box = universe.dimensions[:3]
    positions -= np.floor(positions / box) * box  
    tree = cKDTree(positions, boxsize=box)
    neighbors = tree.query_ball_point(positions, cutoff)
    q6_values = np.zeros(len(positions))
    
    for i, neighs in enumerate(neighbors):
        if len(neighs) <= 1:
            continue    
        q6m = np.zeros(13, dtype=complex)
        
        for j in neighs:
            if i == j:
                continue
            r_ij = positions[j] - positions[i]
            r_ij -= np.round(r_ij / box) * box
            theta = np.arccos(r_ij[2] / np.linalg.norm(r_ij))
            phi = np.arctan2(r_ij[1], r_ij[0])
            for m in range(-6, 7):
                q6m[m + 6] += sph_harm(m, 6, phi, theta)
        q6m /= len(neighs)
        q6_values[i] = np.sqrt((4 * np.pi / 13) * np.sum(np.abs(q6m) ** 2))
    
    return q6_values

for k in range(1, 3):
    for t_0 in t_set:
        for num in range(10):
            source = "../data/Traj/Set_" + str(int(k)) + "/T_" + str(int(num)) + "/"
            traj_file = source + "traj.dcd"
            topo_file = source + "data.0"

            u = md.Universe(topo_file, traj_file, topology_format="DATA", atom_style="id type x y z vx vy vz")
            n_atoms = len(u.atoms)
            order_disorder = np.zeros(n_atoms, dtype="int")

            q6_values = compute_q6(u, frame=t_0)  
            order_disorder = (q6_values > threshold_q6).astype(int)  
            
            save_dir = "../data/order_disorder/set_" + str(int(k)) + "/D_" + str(int(num)) + "/"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            save_path = save_dir + "Y_" + str(int(t_0)) + ".npy"
            np.save(save_path, order_disorder, allow_pickle=True)
        
        print(k, "_", int(t_0))
