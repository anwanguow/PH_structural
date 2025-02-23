import os
import numpy as np
import MDAnalysis as md

source = "traj/"
traj_file = source + "traj.dcd"

u = md.lib.formats.libdcd.DCDFile(traj_file)
traj = u.readframes()[0]

frame_indices = np.arange(0, 101, 1)

output_dir = "distance_matrix/"
os.makedirs(output_dir, exist_ok=True)

for frame in frame_indices:
    pos = traj[frame]

    Dt = np.zeros((len(pos), len(pos)), dtype="float32")

    for i in range(len(pos) - 1):
        vec_i = pos[i]
        for j in range(i + 1, len(pos)):
            vec_j = pos[j]
            vec_ij = vec_i - vec_j
            Dt[i, j] = np.sqrt(
                np.power(vec_ij[0], 2) + np.power(vec_ij[1], 2) + np.power(vec_ij[2], 2)
            )

    Dt = Dt + np.transpose(Dt)

    np.save(os.path.join(output_dir, f"D_{frame}.npy"), Dt, allow_pickle=True)

    print(f"Frame {frame} processed.")
