#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from ripser import Rips
from ripser import ripser
import numpy as np
import collections
import matplotlib.pyplot as plt
collections.Iterable = collections.abc.Iterable

def Makexyzdistance(t):
    element = np.loadtxt(t, dtype=str, usecols=(0,), skiprows=2)
    x = np.loadtxt(t, dtype=float, usecols=(1), skiprows=2)
    y = np.loadtxt(t, dtype=float, usecols=(2), skiprows=2)
    z = np.loadtxt(t, dtype=float, usecols=(3), skiprows=2)

    Distance = np.zeros(shape=(len(x), len(x)))
    for i in range(len(x)):
        for j in range(len(x)):
            Distance[i][j] = np.sqrt(((x[i] - x[j])**2) + ((y[i] - y[j])**2) + ((z[i] - z[j])**2))
    return [Distance, element]

def compute_betti_numbers_at_radius(dgms, radius):
    betti_0 = np.sum((dgms[0][:, 0] <= radius) & (dgms[0][:, 1] > radius))
    betti_1 = np.sum((dgms[1][:, 0] <= radius) & (dgms[1][:, 1] > radius))
    betti_2 = np.sum((dgms[2][:, 0] <= radius) & (dgms[2][:, 1] > radius))
    return betti_0, betti_1, betti_2


set_ = 2
traj_ = 9
frame_ = 900
mksize = 50


file = 'distance_matrix/set_' + str(set_) + '/D_' + str(traj_-1) + '/D_' + str(frame_) + '.npy'
D = np.load(file, allow_pickle=True)

# Set parameters for persistent homology
rips = Rips(maxdim=2)

# Generate persistent homology data
a = ripser(D, distance_matrix=True, maxdim=2)

# Extract H0, H1, and H2 birth-death times
pointsh0 = a['dgms'][0]
pointsh1 = a['dgms'][1]
pointsh2 = a['dgms'][2]

# Define a range of radii
max_death_time = max(np.max(pointsh0[:, 1][np.isfinite(pointsh0[:, 1])]), 
                     np.max(pointsh1[:, 1][np.isfinite(pointsh1[:, 1])]),
                     np.max(pointsh2[:, 1][np.isfinite(pointsh2[:, 1])]))
radii = np.linspace(0, max_death_time, 100)

# Calculate Betti numbers for each radius
betti_0_list = []
betti_1_list = []
betti_2_list = []
for radius in radii:
    betti_0, betti_1, betti_2 = compute_betti_numbers_at_radius(a['dgms'], radius)
    betti_0_list.append(betti_0)
    betti_1_list.append(betti_1)
    betti_2_list.append(betti_2)

# Plot Betti numbers as a function of radius
plt.figure(figsize=(10, 5))
plt.plot(radii, betti_0_list, label='Betti-0')
plt.plot(radii, betti_1_list, label='Betti-1')
plt.plot(radii, betti_2_list, label='Betti-2')
plt.xlabel('Radius')
plt.ylabel('Betti Number')
plt.title('Betti Numbers as a Function of Radius')
plt.legend()
plt.show()

# Plot the persistence diagrams as life-persistence diagrams
plt.figure(figsize=(15, 5))

plt.subplot(131)
plt.scatter(pointsh0[:, 0], pointsh0[:, 1] - pointsh0[:, 0], label='H0')
plt.xlabel('Birth')
plt.ylabel('Persistence')
plt.title('H0 Persistence Diagram')
plt.legend()

plt.subplot(132)
plt.scatter(pointsh1[:, 0], pointsh1[:, 1] - pointsh1[:, 0], label='H1', color='red')
plt.xlabel('Birth')
plt.ylabel('Persistence')
plt.title('H1 Persistence Diagram')
plt.legend()

plt.subplot(133)
plt.scatter(pointsh2[:, 0], pointsh2[:, 1] - pointsh2[:, 0], label='H2', color='green')
plt.xlabel('Birth')
plt.ylabel('Persistence')
plt.title('H2 Persistence Diagram')
plt.legend()

plt.tight_layout()
plt.show()

# Plot all persistence diagrams in one plot with 1:1 aspect ratio
plt.figure(figsize=(6, 6), dpi=300)

plt.scatter(pointsh0[:, 0], pointsh0[:, 1] - pointsh0[:, 0], label='$H_0$', marker='.', color='blue', s=mksize)
plt.scatter(pointsh1[:, 0], pointsh1[:, 1] - pointsh1[:, 0], label='$H_1$', marker='.', color='red', s=mksize)
plt.scatter(pointsh2[:, 0], pointsh2[:, 1] - pointsh2[:, 0], label='$H_2$', marker='.', color='green', s=mksize)
plt.xlabel('Birth', fontsize=20)
plt.ylabel('Persistence', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

# Custom legend with larger points
h0_line = plt.Line2D([0], [0], marker='.', color='blue', markersize=20, linestyle='None', label='$H_0$')
h1_line = plt.Line2D([0], [0], marker='.', color='red', markersize=20, linestyle='None', label='$H_1$')
h2_line = plt.Line2D([0], [0], marker='.', color='green', markersize=20, linestyle='None', label='$H_2$')
plt.legend(handles=[h0_line, h1_line, h2_line], fontsize=20)

# Determine the limits for the axes
all_births = np.concatenate([pointsh0[:, 0], pointsh1[:, 0], pointsh2[:, 0]])
all_persistences = np.concatenate([pointsh0[:, 1] - pointsh0[:, 0], pointsh1[:, 1] - pointsh1[:, 0], pointsh2[:, 1] - pointsh2[:, 0]])

# Filter out NaN and Inf values
all_births = all_births[np.isfinite(all_births)]
all_persistences = all_persistences[np.isfinite(all_persistences)]

max_range = max(np.max(all_births), np.max(all_persistences))

# Expand the limits slightly to ensure points on the border are within the limits
padding = max_range * 0.05  # 5% padding
offset = padding / 2  # Offset to move origin

plt.xlim([0 - offset, max_range + padding - offset])
plt.ylim([0 - offset, max_range + padding - offset])
plt.gca().set_aspect('equal', adjustable='box')

plt.show()
