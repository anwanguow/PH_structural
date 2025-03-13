#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import MDAnalysis as mda
import warnings
from PH import PI_matrix
from scipy.spatial import cKDTree

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

group_ = 1    # {1, 2}
traj_ = 1     # {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
frame_ = 100   # {0, 1, 2, ..., 1000}
r_max_ = 2.0   # {1.1, 1.2, 1.3, ..., 5.1} in article=5.1
particle_ = 50  # {0, 1, ..., 863}  in article, 864 particles
N_ = 40        # in article = \mathfrak{m} = \mathfrak{n} = 40

def apply_pbc(positions, box):
    return (positions + box) % box

def compute_barcode(positions, box, r_max):
    positions = apply_pbc(positions, box)
    tree = cKDTree(positions, boxsize=box)
    pairs = tree.query_pairs(r_max)
    distances = [np.linalg.norm(positions[i] - positions[j]) for i, j in pairs]
    birth = np.zeros(len(distances))
    barcode = np.column_stack((birth, distances))
    return barcode

def get_particle_pi_vector(group, traj, frame, num, r_max, N):
    traj_path = f"../data/Traj/Set_{group}/T_{traj-1}/traj.dcd"
    topo_path = f"../data/Traj/Set_{group}/T_{traj-1}/data.0"
    u = mda.Universe(topo_path, traj_path, topology_format="DATA", atom_style="id type x y z vx vy vz")
    u.trajectory[frame]
    
    box = u.dimensions[:3]
    positions = apply_pbc(u.atoms.positions.copy(), box)
    particle_pos = positions[num]
    
    r_values = np.arange(1.0, r_max, 0.1)
    r_0 = 1.0
    tau = 0.57
    feature_vector = None
    
    tree = cKDTree(positions, boxsize=box)
    
    with open("PI_vec_particle.txt", "w") as f:
        for r in r_values:
            neighbor_indices = tree.query_ball_point(particle_pos, r)
            neighbor_positions = positions[neighbor_indices]
            
            for dim in range(3):
                if np.any(neighbor_positions[:, dim] - particle_pos[dim] > box[dim] / 2):
                    neighbor_positions = np.vstack((neighbor_positions, neighbor_positions - box[dim]))
                if np.any(particle_pos[dim] - neighbor_positions[:, dim] > box[dim] / 2):
                    neighbor_positions = np.vstack((neighbor_positions, neighbor_positions + box[dim]))
            
            if len(neighbor_positions) < 2:
                continue
            
            barcode = compute_barcode(neighbor_positions, box, r)
            PI = PI_matrix(barcode, pixelx=N, pixely=N, myspread=0.01, myspecs={"maxBD": 1.5, "minBD":-0.1}, showplot=False)
            weight = np.exp(-tau * (r - r_0))
            weighted_PI = PI * weight
            if feature_vector is None:
                feature_vector = weighted_PI.flatten()
            else:
                feature_vector += weighted_PI.flatten()
        
        feature_vector = feature_vector.reshape(-1, 1)
        f.write(f"\\textbf{{I}}_{{{num}}}^{{(P)}} = \\\n")
        np.savetxt(f, feature_vector, fmt="%.6e")
    
    print(f"\\textbf{{I}}_{{{num}}}^{{(P)}} =")
    print(feature_vector)
    
get_particle_pi_vector(group_, traj_, frame_, particle_, r_max_, N_)

