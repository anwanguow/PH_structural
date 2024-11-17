import numpy as np
import os
import pickle
from PointCloud import get_neighbor
from PH import PH_barcode

traj_index = 0

set_dirs = ["data/distance_matrix"]
output_dirs = ["PH"]
frames = [110]
r_values = np.arange(1.0, 5.1, 0.1)

for set_index, set_dir in enumerate(set_dirs):
    traj_dir = os.path.join(set_dir, f"")
    output_traj_dir = os.path.join(output_dirs[set_index], f"")
    os.makedirs(output_traj_dir, exist_ok=True)
    
    for frame in frames:
        distance_matrix_path = os.path.join(traj_dir, f"D_{frame}.npy")
        D = np.load(distance_matrix_path)
        num_particles = D.shape[0]        
        for r in r_values:
            barcodes = []
            for i in range(num_particles):
                neighbor, indices = get_neighbor(D, i, r=r)
                barcode = PH_barcode(neighbor)
                barcodes.append(barcode)
            output_path = os.path.join(output_traj_dir, f"D_{frame}_{r:.1f}.pkl")
            with open(output_path, 'wb') as f:
                pickle.dump(barcodes, f)

