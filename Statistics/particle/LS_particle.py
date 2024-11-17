#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
from scipy.stats import entropy
import matplotlib.pyplot as plt
import seaborn as sns

epsilon = 2.5

def calculate_separation_index_per_sample(row):
    if row['dimension'] == r'$H_1$':
        h1_persistence = row['death_times'] - row['birth_times']
        mean_h1 = np.mean(h1_persistence)
        std_h1 = np.std(h1_persistence)
        return mean_h1, std_h1
    elif row['dimension'] == r'$H_2$':
        h2_persistence = row['death_times'] - row['birth_times']
        mean_h2 = np.mean(h2_persistence)
        std_h2 = np.std(h2_persistence)
        return mean_h2, std_h2
    return None, None

def compute_persistent_homology_features(barcode_data):
    stats = []
    label = barcode_data['label']
    for dim, latex_dim in zip(['H0', 'H1', 'H2'], [r'$H_0$', r'$H_1$', r'$H_2$']):
        diagrams = barcode_data[dim]
        if len(diagrams) == 0:
            continue
        lengths = diagrams[:, 1] - diagrams[:, 0]
        lengths = lengths[np.isfinite(lengths)]
        stats.append({
            "dimension": latex_dim,
            "num_barcodes": len(lengths),
            "max_length": np.max(lengths),
            "mean_length": np.mean(lengths),
            "persistent_entropy": entropy(lengths / np.sum(lengths)),
            "max_birth": np.max(diagrams[:, 0]),
            "mean_death": np.mean(diagrams[:, 1][np.isfinite(diagrams[:, 1])]),
            "lifetime": lengths,
            "label": label,
            "birth_times": diagrams[:, 0],
            "death_times": diagrams[:, 1]
        })
    return stats

barcode_dir = "cut_r_" + str(epsilon) + "/barcodes"
all_features = []

for barcode_file in os.listdir(barcode_dir):
    if barcode_file.endswith(".npy"):
        barcodes = np.load(os.path.join(barcode_dir, barcode_file), allow_pickle=True)
        for barcode_data in barcodes:
            features = compute_persistent_homology_features(barcode_data)
            all_features.extend(features)

features_df = pd.DataFrame(all_features)

separation_indices = []

for idx, row in features_df.iterrows():
    if row['dimension'] == r'$H_1$':
        mean_h1, std_h1 = calculate_separation_index_per_sample(row)
        if idx + 1 < len(features_df):
            next_row = features_df.iloc[idx + 1]
            if next_row['dimension'] == r'$H_2$':
                mean_h2, std_h2 = calculate_separation_index_per_sample(next_row)
                if mean_h1 is not None and mean_h2 is not None:
                    separation_index = np.abs(mean_h1 - mean_h2) / (std_h1 + std_h2)
                    separation_indices.append({
                        'label': row['label'],
                        'separation_index': separation_index
                    })

separation_df = pd.DataFrame(separation_indices)
separation_df = separation_df[separation_df['separation_index'] <= epsilon]
label_map = {0: 'Hard', 1: 'Soft'}
features_df['label_name'] = features_df['label'].map(label_map)

color_map = {r'$H_0$': '#4B0082', r'$H_1$': '#006400', r'$H_2$': '#FFD700'}

plt.figure(figsize=(6, 6), dpi=300)
exploded_df = features_df.explode('lifetime')
sns.boxplot(data=exploded_df, x='label_name', y='lifetime', hue='dimension', palette=color_map)
plt.xlabel('Class', fontsize=20)
plt.ylabel('Barcode Lifespan', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(fontsize=16, loc='best', handlelength=1.2)
plt.savefig('Lifespan.png', bbox_inches='tight', pad_inches=0.1)
plt.show()
