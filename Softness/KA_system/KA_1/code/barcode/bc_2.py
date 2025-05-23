import os
import numpy as np
from ripser import ripser
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
distance_matrix_path_template = "data/distance_matrix/D_{F}.npy"

file_paths = []
sources = []

for p in range(1, 11): # 310 - 400
    F = 310 + (p - 1) * 10
    file_path = distance_matrix_path_template.format(F=F)
    file_paths.append(file_path)
    sources.append(f"bc")

def compute_and_save_barcodes(file_path, source, output_dir):
    try:
        points = np.load(file_path)
        diagrams = ripser(points, maxdim=2, distance_matrix=True)['dgms']
        barcode_data = {'H0': diagrams[0], 'H1': diagrams[1], 'H2': diagrams[2], 'path': file_path}
        output_file = os.path.join(output_dir, f"{source}_{os.path.basename(file_path).replace('.npy', '')}_barcodes.npy")
        np.save(output_file, barcode_data)
        logging.info(f"Successfully saved barcodes for: {file_path}")
    except Exception as e:
        logging.error(f"Error processing file: {file_path}, {e}")

output_dir = "barcodes"
os.makedirs(output_dir, exist_ok=True)

for file_path, source in zip(file_paths, sources):
    compute_and_save_barcodes(file_path, source, output_dir)
