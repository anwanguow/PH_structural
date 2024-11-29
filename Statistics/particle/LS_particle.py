#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

epsilon = 2.5

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
            "lifetime": lengths,
            "label": label,
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
