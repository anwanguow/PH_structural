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


set_n = np.array([1,2])
num = np.array([0,1,2,3,4,5,6,7,8,9])
time_list = [100] + list(range(110, 910, 10))

tau = 0.57

for set_ in set_n:
    for i in num:    
        for t in time_list:
            file_D = "PI/decay/set_" + str(int(set_)) + "/D_" + str(int(i)) + "/PI_" + str(int(t)) + "_tau_" + str(tau) + ".npy";
            D = np.load(file_D, allow_pickle=True);
            new_matrix = normalize_columns(D);
            df = pd.DataFrame(new_matrix);
            output_name = "matlab/database/decay/set_" + str(int(set_)) + "/D_" + str(int(i)) + "/X_t_" + str(t) + "_tau_" + str(tau) + ".csv"
            df.to_csv(output_name, index=False, header=False)
            print(set_, "_", i, "_", t)
        

