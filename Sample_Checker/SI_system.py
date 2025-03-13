#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import MDAnalysis as mda
import pandas as pd
from ripser import ripser

group_ = 1    # {1, 2}
traj_ = 1     # {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
frame_ = 100   # {0, 1, 2, ..., 1000}


def compute_persistent_homology(positions):
    result = ripser(positions, maxdim=2)
    return {
        "H0": result['dgms'][0],
        "H1": result['dgms'][1],
        "H2": result['dgms'][2] if len(result['dgms']) > 2 else np.array([])
    }

def compute_persistent_homology_features(barcode_data):
    stats = []
    for dim, diagrams in barcode_data.items():
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
    h1_persistence = df[df['dimension'] == 'H1'].explode('lifetime')['lifetime'].dropna()
    h2_persistence = df[df['dimension'] == 'H2'].explode('lifetime')['lifetime'].dropna()
    return np.abs(h1_persistence.mean() - h2_persistence.mean()) / (h1_persistence.std() + h2_persistence.std())

def compute_system_separation_index(group, traj, frame):
    traj_path = f"../data/Traj/Set_{group}/T_{traj-1}/traj.dcd"
    topo_path = f"../data/Traj/Set_{group}/T_{traj-1}/data.0"
    
    u = mda.Universe(topo_path, traj_path, topology_format="DATA", atom_style="id type x y z vx vy vz")
    u.trajectory[frame]
    
    positions = u.atoms.positions.copy()
    
    barcode_data = compute_persistent_homology(positions)
    features = compute_persistent_homology_features(barcode_data)
    df = pd.DataFrame(features)
    
    separation_index = calculate_separation_index(df)
    print(f"Separation Index for system: {separation_index:.4f}")

compute_system_separation_index(group_, traj_, frame_)
