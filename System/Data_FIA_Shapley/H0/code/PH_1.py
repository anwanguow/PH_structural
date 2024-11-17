import numpy as np
import os
import pickle
from PH import PH_barcode_h0

set_dirs = ["../../data/distance_matrix/set_1", "../../data/distance_matrix/set_2"]
output_dirs = ["../PH/barcode/set_1", "../PH/barcode/set_2"]
frames = [200] + list(range(225, 300, 25))

for set_index, set_dir in enumerate(set_dirs):
    for traj_index in range(10):
        traj_dir = os.path.join(set_dir, f"D_{traj_index}")
        output_traj_dir = os.path.join(output_dirs[set_index], f"D_{traj_index}")
        os.makedirs(output_traj_dir, exist_ok=True)
        
        for frame in frames:
            distance_matrix_path = os.path.join(traj_dir, f"D_{frame}.npy")
            D = np.load(distance_matrix_path)

            barcode = PH_barcode_h0(D)
            
            output_path = os.path.join(output_traj_dir, f"D_{frame}.pkl")
            with open(output_path, 'wb') as f:
                pickle.dump(barcode, f)
