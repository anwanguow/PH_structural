#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# compute the degree of seperation

import os
import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def compute_persistent_homology_features(barcode_data):
    stats = []
    for dim in ['H0', 'H1', 'H2']:
        diagrams = barcode_data[dim]
        if len(diagrams) == 0:
            continue
        lengths = diagrams[:, 1] - diagrams[:, 0]
        lengths = lengths[np.isfinite(lengths)]
        stats.append({
            "dimension": dim,
            "lifetime": lengths,
            "birth_times": diagrams[:, 0],
            "death_times": diagrams[:, 1]
        })
    return stats

def load_barcode_data(file_path):
    barcode_data = np.load(file_path, allow_pickle=True).item()
    features = compute_persistent_homology_features(barcode_data)
    return pd.DataFrame(features)

def calculate_separation_index(df):
    h1_persistence = df[df['dimension'] == 'H1'].apply(lambda row: row['death_times'] - row['birth_times'], axis=1).explode().dropna()
    h2_persistence = df[df['dimension'] == 'H2'].apply(lambda row: row['death_times'] - row['birth_times'], axis=1).explode().dropna()
    
    if len(h1_persistence) > 0 and len(h2_persistence) > 0:
        mean_h1 = np.mean(h1_persistence)
        mean_h2 = np.mean(h2_persistence)
        std_h1 = np.std(h1_persistence)
        std_h2 = np.std(h2_persistence)
        separation_index = np.abs(mean_h1 - mean_h2) / (std_h1 + std_h2)
        return separation_index
    else:
        return None

def main():
    separation_indices = []
    
    for frame in range(100, 901, 10):
        file_path = f"barcodes/bc_D_{frame}_barcodes.npy"
        features_df = load_barcode_data(file_path)
        separation_index = calculate_separation_index(features_df)
        
        if separation_index is not None:
            separation_indices.append(separation_index)
        else:
            separation_indices.append(np.nan)
    
    output_dir = "sep"
    os.makedirs(output_dir, exist_ok=True)
    sep_output_file = os.path.join(output_dir, f"sep.npy")
    np.save(sep_output_file, np.array(separation_indices))
    logging.info(f"Separation indices saved to {sep_output_file}")

main()
