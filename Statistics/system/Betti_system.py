import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def compute_betti_numbers_over_filtration(barcode_data, min_filtration=0, max_filtration=2, step=0.01):
    betti_numbers = {'label': barcode_data['label']}
    filtration_values = np.arange(min_filtration, max_filtration + step, step)
    for dim in ['H0', 'H1', 'H2']:
        diagrams = barcode_data[dim]
        if len(diagrams) == 0:
            betti_numbers[dim] = np.zeros_like(filtration_values)
            continue
        betti_counts = []
        for t in filtration_values:
            betti_counts.append(np.sum((diagrams[:, 0] <= t) & (diagrams[:, 1] > t)))
        betti_numbers[dim] = betti_counts
    return betti_numbers, filtration_values

barcode_dir = "barcodes"
all_betti_numbers = []

for barcode_file in os.listdir(barcode_dir):
    if barcode_file.endswith(".npy"):
        barcode_data = np.load(os.path.join(barcode_dir, barcode_file), allow_pickle=True).item()
        betti_numbers, filtration_values = compute_betti_numbers_over_filtration(barcode_data)
        all_betti_numbers.append(betti_numbers)

betti_df_list = []
for betti_numbers in all_betti_numbers:
    label = betti_numbers.pop('label')
    df = pd.DataFrame(betti_numbers)
    df['filtration'] = filtration_values
    df['label'] = label
    betti_df_list.append(df)
betti_df = pd.concat(betti_df_list)

sns.set(style="white")

def plot_combined_betti_numbers(df, filename):
    plt.figure(figsize=(6, 6), dpi=300)
    
    color_map = {'0': '#1f77b4', '1': '#ff7f0e', '2': '#2ca02c'}
    style_map = {'H0': '-', 'H1': '-.', 'H2': ':'}
    label_map = {'0': 'Liquid', '1': 'Crystal', '2': 'Amorphous'}
    
    legend_labels = {
        '0': { 'H0': r'$\beta_{0,liq}$', 'H1': r'$\beta_{1,liq}$', 'H2': r'$\beta_{2,liq}$' },
        '1': { 'H0': r'$\beta_{0,cry}$', 'H1': r'$\beta_{1,cry}$', 'H2': r'$\beta_{2,cry}$' },
        '2': { 'H0': r'$\beta_{0,amo}$', 'H1': r'$\beta_{1,amo}$', 'H2': r'$\beta_{2,amo}$' }
    }
    
    for dim in ['H0', 'H1', 'H2']:
        for label in df['label'].unique():
            label_str = str(label)
            subset = df[df['label'] == label]
            sns.lineplot(data=subset, x='filtration', y=dim, color=color_map[label_str], linestyle=style_map[dim], linewidth=2.5, label=legend_labels[label_str][dim])
    
    plt.xlabel('Filtration Value $\\epsilon$', fontsize=21)
    plt.ylabel('Betti Number $\\beta$', fontsize=21)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid(False)
    
    plt.tick_params(axis='x', which='both', direction='in', top=False, bottom=True, labelsize=21)
    plt.tick_params(axis='y', which='both', direction='in', left=True, right=False, labelsize=21)
    
    handles, labels = plt.gca().get_legend_handles_labels()
    unique_labels = []
    unique_handles = []
    for handle, label in zip(handles, labels):
        if label not in unique_labels:
            unique_labels.append(label)
            unique_handles.append(handle)
    
    plt.legend(handles=unique_handles, labels=unique_labels, title='', loc='best', fontsize=21)
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

plot_combined_betti_numbers(betti_df, 'Betti_numbers.png')
