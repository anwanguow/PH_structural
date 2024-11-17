import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

pi_dirs = ["PI/set_1", "PI/set_2"]
label_files = ["label/label_set_1.csv", "label/label_set_2.csv"]
output_files = ["dataset/dataset_set_1.csv", "dataset/dataset_set_2.csv", "dataset/dataset_combined.csv"]
frames = [100] + list(range(125, 925, 25))

labels = []
for label_file in label_files:
    labels.append(np.loadtxt(label_file, delimiter=','))

def generate_feature_matrix(pi_dirs, frames, labels):
    feature_matrix = []
    for set_index, pi_dir in enumerate(pi_dirs):
        for traj_index in range(10):
            traj_dir = os.path.join(pi_dir, f"D_{traj_index}")
            for frame_index, frame in enumerate(frames):
                pi_path = os.path.join(traj_dir, f"D_{frame}.npy")
                pi = np.load(pi_path).flatten()
                label = labels[set_index][traj_index, frame_index]
                feature_matrix.append(np.append(pi, label))
    return np.array(feature_matrix)

feature_matrix_set_1 = generate_feature_matrix([pi_dirs[0]], frames, [labels[0]])
feature_matrix_set_2 = generate_feature_matrix([pi_dirs[1]], frames, [labels[1]])
feature_matrix_combined = np.vstack((feature_matrix_set_1, feature_matrix_set_2))

scaler = MinMaxScaler()

X_combined = feature_matrix_combined[:, :-1]
y_combined = feature_matrix_combined[:, -1]

X_combined_normalized = scaler.fit_transform(X_combined)

feature_matrix_combined_normalized = np.hstack((X_combined_normalized, y_combined.reshape(-1, 1)))

feature_matrix_set_1_normalized = feature_matrix_combined_normalized[:len(feature_matrix_set_1)]
feature_matrix_set_2_normalized = feature_matrix_combined_normalized[len(feature_matrix_set_1):]

np.savetxt(output_files[0], feature_matrix_set_1_normalized, delimiter=',', fmt='%g')
np.savetxt(output_files[1], feature_matrix_set_2_normalized, delimiter=',', fmt='%g')
np.savetxt(output_files[2], feature_matrix_combined_normalized, delimiter=',', fmt='%g')
