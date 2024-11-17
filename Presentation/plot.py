#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from ripser import ripser
from PI import PI

xyz_file_path = 'niclosamide.xyz'
myspread = 0.0005
radius = 3.0

def read_xyz(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        atom_count = int(lines[0].strip())
        atoms = []
        for line in lines[2:2+atom_count]:
            parts = line.split()
            element = parts[0]
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            atoms.append((element, x, y, z))
        return atoms

def center_atoms(atoms):
    coords = np.array([(x, y, z) for _, x, y, z in atoms])
    centroid = np.mean(coords, axis=0)
    centered_atoms = [(element, x - centroid[0], y - centroid[1], z - centroid[2]) for element, x, y, z in atoms]
    return centered_atoms

def compute_persistence_diagrams(atoms, max_radius):
    coords = np.array([(x, y, z) for _, x, y, z in atoms])
    diagrams = ripser(coords, maxdim=2, thresh=max_radius)['dgms']
    return diagrams

def plot_persistence_diagrams(diagrams, ax, max_radius, offset=0.05, show_legend=False):
    colors = ['#ff9999', '#66b3ff', '#99ff99']
    labels = [r'$H_0$', r'$H_1$', r'$H_2$']
    for i, diagram in enumerate(diagrams):
        if len(diagram) > 0:
            births = diagram[:, 0]
            deaths = diagram[:, 1]
            lifetimes = deaths - births
            is_alive = np.isinf(deaths)
            deaths[is_alive] = max_radius
            lifetimes = deaths - births
            ax.scatter(births, lifetimes, label=labels[i], color=colors[i], edgecolor='k', alpha=0.7, s=400)
    ax.set_xlim(-max_radius * offset, max_radius * 1.1)
    ax.set_ylim(-max_radius * offset, max_radius * 1.1)
    ax.set_xlabel('Birth', fontsize=32)
    ax.set_ylabel('Lifetime', fontsize=32)
    ax.tick_params(axis='both', which='major', labelsize=24)
    if show_legend:
        ax.legend(fontsize=24)

def plot_molecule_with_radii_and_persistence(atoms, radius, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=300)
    coords = np.array([(x, y, z) for _, x, y, z in atoms])
    diagrams = compute_persistence_diagrams(atoms, radius)
    ax = axes[0]
    total_bars = 0
    for dim, diagram in enumerate(diagrams):
        if len(diagram) > 0:
            sorted_diagram = diagram[np.argsort(diagram[:, 0])]
            for j, (birth, death) in enumerate(sorted_diagram):
                lifetime = death - birth
                if np.isinf(death):
                    death = radius
                    lifetime = death - birth
                ax.plot([birth, death], 
                        [total_bars + j + 0.5, total_bars + j + 0.5], 
                        color=['#ff9999', '#66b3ff', '#99ff99'][dim], lw=8, alpha=0.8)
            total_bars += len(sorted_diagram)
    ax.set_xlim(-radius * 0.05, radius * 1.1)
    ax.set_ylim(total_bars + 1, -0.5)
    ax.yaxis.set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('k')
    ax.set_xlabel(r'Filtration Variable $\epsilon$', fontsize=32)
    ax.set_ylabel('Lifetime (Bars)', fontsize=32)
    ax.tick_params(axis='both', which='major', labelsize=24)
    ax = axes[1]
    plot_persistence_diagrams(diagrams, ax, radius, offset=0.05, show_legend=True)
    pim = PI(pixels=[10, 10], spread=myspread, specs={"maxBD": radius, "minBD": 0}, kernel_type="gaussian", verbose=False)
    imgs = pim.transform(diagrams)
    ax = axes[2]
    pim.show(imgs, ax=ax)
    ax.set_xlabel('X-axis (Image)', fontsize=32)
    ax.set_ylabel('Y-axis (Image)', fontsize=32)
    ax.tick_params(axis='both', which='major', labelsize=24)
    plt.tight_layout(pad=1.0)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

atoms = read_xyz(xyz_file_path)
centered_atoms = center_atoms(atoms)
save_path = 'plot.png'
plot_molecule_with_radii_and_persistence(centered_atoms, radius, save_path)
