#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

def normalize_columns(matrix):
    normalized_matrix = np.zeros_like(matrix)
    num_columns = matrix.shape[1]
    for i in range(num_columns - 1):
        col_min = np.min(matrix[:, i])
        col_max = np.max(matrix[:, i])
        if col_max - col_min > 0:
            normalized_matrix[:, i] = (matrix[:, i] - col_min) / (col_max - col_min)
        else:
            normalized_matrix[:, i] = 0
    normalized_matrix[:, -1] = matrix[:, -1]    
    return normalized_matrix


time_list = [100] + list(range(110, 910, 10))

for t in time_list:
    file_D = "PI/PI_" + str(int(t)) + "_tau_0.57.npy";
    D = np.load(file_D, allow_pickle=True);
    new_matrix = normalize_columns(D);
    df = pd.DataFrame(new_matrix);
    output_name = "matlab/data/X_t_" + str(t) + ".csv"
    df.to_csv(output_name, index=False, header=False)
    print(t)

