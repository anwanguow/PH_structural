#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_separation_index_per_sample(row):
    if row['dimension'] == 'H1':
        h1_persistence = row['death_times'] - row['birth_times']
        mean_h1 = np.mean(h1_persistence)
        std_h1 = np.std(h1_persistence)
        return mean_h1, std_h1
    elif row['dimension'] == 'H2':
        h2_persistence = row['death_times'] - row['birth_times']
        mean_h2 = np.mean(h2_persistence)
        std_h2 = np.std(h2_persistence)
        return mean_h2, std_h2
    return None, None

def compute_persistent_homology_features(barcode_data):
    stats = []
    label = barcode_data['label']
    for dim in ['H1', 'H2']:
        diagrams = barcode_data[dim]
        if len(diagrams) == 0:
            continue
        stats.append({
            "dimension": dim,
            "birth_times": diagrams[:, 0],
            "death_times": diagrams[:, 1],
            "label": label
        })
    return stats

radii = [1.5, 2.5, 3.5, 4.5]
all_features = []

for radius in radii:
    barcode_dir = f"cut_r_{radius}/barcodes"
    for barcode_file in os.listdir(barcode_dir):
        if barcode_file.endswith(".npy"):
            barcodes = np.load(os.path.join(barcode_dir, barcode_file), allow_pickle=True)
            for barcode_data in barcodes:
                features = compute_persistent_homology_features(barcode_data)
                for feature in features:
                    feature['radius'] = radius
                all_features.extend(features)

features_df = pd.DataFrame(all_features)
separation_indices = []

for idx, row in features_df.iterrows():
    if row['dimension'] == 'H1':
        mean_h1, std_h1 = calculate_separation_index_per_sample(row)
        next_row = features_df.iloc[idx + 1]
        if next_row['dimension'] == 'H2':
            mean_h2, std_h2 = calculate_separation_index_per_sample(next_row)
            if mean_h1 is not None and mean_h2 is not None:
                std_sum = std_h1 + std_h2
                if std_sum == 0:
                    separation_index = np.inf
                else:
                    separation_index = np.abs(mean_h1 - mean_h2) / std_sum
                separation_indices.append({
                    'label': row['label'],
                    'separation_index': separation_index,
                    'radius': row['radius']
                })

separation_df = pd.DataFrame(separation_indices)
separation_df = separation_df[separation_df['separation_index'] <= 5]
label_map = {0: 'Hard', 1: 'Soft'}
separation_df['label_name'] = separation_df['label'].map(label_map)
separation_df['radius'] = separation_df['radius'].astype(str)

palette = {'Hard': 'salmon', 'Soft': 'lightblue'}

plt.figure(figsize=(6, 6), dpi=300)
sns.boxplot(x='radius', y='separation_index', hue='label_name', data=separation_df, palette=palette, linewidth=2.5)
plt.xlabel('Neighborhood Radius', fontsize=20)
plt.ylabel('Separation Index', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(handles=handles, labels=labels, title='', fontsize=20, handlelength=1, handletextpad=0.4)

plt.savefig('SI_neighborhoods.png')
plt.show()
