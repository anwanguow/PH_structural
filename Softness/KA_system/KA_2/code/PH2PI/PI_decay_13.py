import numpy as np
import os
import pickle
from PH import PI_matrix

set_dirs = ["PH"]
output_dirs = ["PI"]
frames = np.array([460,470,480])
r_values = np.arange(1.0, 5.1, 0.1)
r_0 = 1.0

pixelx = 40
pixely = 40
myspread = 0.01
tau = 0.57
max_bd = 1.5

for set_index, set_dir in enumerate(set_dirs):
    for traj_index in range(1):
        traj_dir = os.path.join(set_dir, f"")
        output_traj_dir = os.path.join(output_dirs[set_index], f"")
        os.makedirs(output_traj_dir, exist_ok=True)
        
        for frame in frames:
            feature_matrix = []
            for r in r_values:
                pkl_path = os.path.join(traj_dir, f"D_{frame}_{r:.1f}.pkl")
                with open(pkl_path, 'rb') as f:
                    barcodes = pickle.load(f)
                particle_PI_matrices = []
                
                for barcode in barcodes:
                    PI = PI_matrix(barcode, pixelx=pixelx, pixely=pixely, myspread=myspread, myspecs={"maxBD": max_bd, "minBD":-0.1}, showplot=False)
                    particle_PI_matrices.append(PI)
                
                for i, PI in enumerate(particle_PI_matrices):
                    weight = np.exp(-tau * (r - r_0))
                    weighted_PI = PI * weight
                    if len(feature_matrix) <= i:
                        feature_matrix.append(weighted_PI.flatten())
                    else:
                        feature_matrix[i] += weighted_PI.flatten()

            feature_matrix = np.array(feature_matrix)
            output_path = os.path.join(output_traj_dir, f"PI_{frame}_tau_{tau}.npy")
            np.save(output_path, feature_matrix)
