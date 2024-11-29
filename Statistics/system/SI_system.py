#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def compute_relevant_features(barcode_data):
    stats = []
    label = barcode_data['label']
    for dim in ['H1', 'H2']:
        diagrams = barcode_data[dim]
        if len(diagrams) == 0:
            continue
        lengths = diagrams[:, 1] - diagrams[:, 0]
        lengths = lengths[np.isfinite(lengths)]
        stats.append({
            "dimension": dim,
            "label": label,
            "birth_times": diagrams[:, 0],
            "death_times": diagrams[:, 1]
        })
    return stats

barcode_dir = "barcodes"
all_features = []

for barcode_file in os.listdir(barcode_dir):
    if barcode_file.endswith(".npy"):
        barcode_data = np.load(os.path.join(barcode_dir, barcode_file), allow_pickle=True).item()
        features = compute_relevant_features(barcode_data)
        all_features.extend(features)

features_df = pd.DataFrame(all_features)

label_map = {0: 'liquid', 1: 'crystal', 2: 'amorphous'}
features_df['label_name'] = features_df['label'].map(label_map)

def calculate_separation_index(row):
    persistence = row['death_times'] - row['birth_times']
    mean_persistence = np.mean(persistence)
    std_persistence = np.std(persistence)
    return mean_persistence, std_persistence

separation_indices = []

for idx, row in features_df.iterrows():
    if row['dimension'] == 'H1':
        mean_h1, std_h1 = calculate_separation_index(row)
        next_row = features_df.iloc[idx + 1]
        if next_row['dimension'] == 'H2':
            mean_h2, std_h2 = calculate_separation_index(next_row)
            if mean_h1 is not None and mean_h2 is not None:
                separation_index = np.abs(mean_h1 - mean_h2) / (std_h1 + std_h2)
                separation_indices.append({
                    'label': row['label'],
                    'label_name': row['label_name'],
                    'separation_index': separation_index
                })

separation_df = pd.DataFrame(separation_indices)

plt.figure(figsize=(6, 6), dpi=300)
sns.boxplot(
    x='label_name', y='separation_index', data=separation_df,
    hue='label_name',
    dodge=False,
    palette={'liquid': '#ADD8E6', 'crystal': '#006400', 'amorphous': '#FFD700'},
    linewidth=2.5
)
if plt.gca().get_legend() is not None:
    plt.gca().get_legend().remove()
plt.xlabel('Class', fontsize=20)
plt.ylabel('Separation Index', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks([0, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8], fontsize=20)

plt.tight_layout()
plt.savefig('SI_system.png', bbox_inches='tight', dpi=300)
plt.show()
