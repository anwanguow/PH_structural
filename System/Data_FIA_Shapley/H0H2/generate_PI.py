#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import os
import pickle
from PH import PI_matrix

pixelx = 40
pixely = 40
myspread = 0.01
max_bd = 1.5

barcode_dirs = ["PH/barcode/set_1", "PH/barcode/set_2"]
output_dirs = ["PI/set_1", "PI/set_2"]
frames = [100] + list(range(125, 925, 25))

for set_index, barcode_dir in enumerate(barcode_dirs):
    for traj_index in range(10):
        traj_dir = os.path.join(barcode_dir, f"D_{traj_index}")
        output_traj_dir = os.path.join(output_dirs[set_index], f"D_{traj_index}")
        os.makedirs(output_traj_dir, exist_ok=True)
        
        for frame in frames:
            barcode_path = os.path.join(traj_dir, f"D_{frame}.pkl")
            with open(barcode_path, 'rb') as f:
                barcode = pickle.load(f)
            
            PI = PI_matrix(barcode, pixelx=pixelx, pixely=pixely, myspread=myspread, myspecs={"maxBD": max_bd, "minBD": -0.1}, showplot=False)
            output_path = os.path.join(output_traj_dir, f"D_{frame}.npy")
            np.save(output_path, PI)

