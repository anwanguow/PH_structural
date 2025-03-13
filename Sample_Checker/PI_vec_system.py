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
N_ = 40        # in article = \mathfrak{m} = \mathfrak{n} = 40

def compute_barcode(positions):
    tree = cKDTree(positions)
    r_max = np.max(tree.query(positions, k=2)[0][:, 1])
    pairs = tree.query_pairs(r_max)
    distances = [np.linalg.norm(positions[i] - positions[j]) for i, j in pairs]
    birth = np.zeros(len(distances))
    return np.column_stack((birth, distances))

def get_system_pi_vector(group, traj, frame, N):
    traj_path = f"../data/Traj/Set_{group}/T_{traj-1}/traj.dcd"
    topo_path = f"../data/Traj/Set_{group}/T_{traj-1}/data.0"
    u = mda.Universe(topo_path, traj_path, topology_format="DATA", atom_style="id type x y z vx vy vz")
    u.trajectory[frame]
    positions = u.atoms.positions.copy()
    barcode = compute_barcode(positions)
    PI = PI_matrix(barcode, pixelx=N, pixely=N, myspread=0.01, myspecs={"maxBD": 1.5, "minBD": -0.1}, showplot=False)
    pi_vector = PI.flatten().reshape(-1, 1)
    np.savetxt("PI_vec_system.txt", pi_vector, fmt="%.6e")
    print("System PI_vec:")
    print(pi_vector)
    
get_system_pi_vector(group_, traj_, frame_, N_)

