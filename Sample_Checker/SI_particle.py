#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import MDAnalysis as mda
import warnings
import pandas as pd
from ripser import ripser
from scipy.spatial import cKDTree

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

group_ = 1    # {1, 2}
traj_ = 1     # {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
frame_ = 100   # {0, 1, 2, ..., 1000}
r_fixed = 2.5   # {1.5, 2.5, 3.5, 4.5} in article
particle_ = 50  # {0, 1, ..., 863}  in article, 864 particles

def compute_persistent_homology(positions):
    if len(positions) < 2:
        return {"H0": np.array([]), "H1": np.array([]), "H2": np.array([])}
    result = ripser(positions, maxdim=2)
    return {
        "H0": result['dgms'][0],
        "H1": result['dgms'][1],
        "H2": result['dgms'][2] if len(result['dgms']) > 2 else np.array([])
    }

def compute_persistent_homology_features(barcode_data):
    stats = []
    for dim, diagrams in barcode_data.items():
        if diagrams.size == 0:
            continue
        lifetimes = diagrams[:, 1] - diagrams[:, 0]
        lifetimes = lifetimes[np.isfinite(lifetimes)]
        stats.append({
            "dimension": dim, 
            "lifetime": lifetimes, 
            "birth_times": diagrams[:, 0], 
            "death_times": diagrams[:, 1]
        })
    return stats

def calculate_separation_index(df):
    if df.empty:
        return None
    h1_persistence = df[df['dimension'] == 'H1'].explode('lifetime')['lifetime'].dropna()
    h2_persistence = df[df['dimension'] == 'H2'].explode('lifetime')['lifetime'].dropna()
    if h1_persistence.empty or h2_persistence.empty:
        return None
    return np.abs(h1_persistence.mean() - h2_persistence.mean()) / (h1_persistence.std() + h2_persistence.std())

def get_particle_separation_index(group, traj, frame, num, r):
    traj_path = f"../data/Traj/Set_{group}/T_{traj-1}/traj.dcd"
    topo_path = f"../data/Traj/Set_{group}/T_{traj-1}/data.0"
    u = mda.Universe(topo_path, traj_path, topology_format="DATA", atom_style="id type x y z vx vy vz")
    u.trajectory[frame]
    
    positions = np.mod(u.atoms.positions.copy(), u.dimensions[:3])
    particle_pos = positions[num]
    
    tree = cKDTree(positions, boxsize=u.dimensions[:3])
    neighbor_indices = tree.query_ball_point(particle_pos, r)
    neighbor_positions = positions[neighbor_indices]
    
    if len(neighbor_positions) < 2:
        print(f"Separation Index for r={r}: Not enough neighbors")
        return
    
    barcode_data = compute_persistent_homology(neighbor_positions)
    features = compute_persistent_homology_features(barcode_data)
    df = pd.DataFrame(features)
    
    separation_index = calculate_separation_index(df)
    if separation_index is None:
        print(f"Separation Index for r={r}: Not enough data (H1 or H2 missing)")
    else:
        print(f"Separation Index for r={r}: {separation_index:.4f}")

get_particle_separation_index(group_, traj_, frame_, particle_, r_fixed)

