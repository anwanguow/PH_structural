#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from ripser import ripser
import numpy as np
import matplotlib.pyplot as plt

set_ = 1
traj_ = 5
frame_ = 900
mksize = 80
markersize_ = 30
ft_size = 30

file = '../../data/distance_matrix/set_' + str(set_) + '/D_' + str(traj_-1) + '/D_' + str(frame_) + '.npy'
D = np.load(file, allow_pickle=True)

a = ripser(D, distance_matrix=True, maxdim=2)

pointsh0 = a['dgms'][0]
pointsh1 = a['dgms'][1]
pointsh2 = a['dgms'][2]

plt.figure(figsize=(6, 6), dpi=300)

plt.scatter(pointsh0[:, 0], pointsh0[:, 1] - pointsh0[:, 0], label='$H_0$', marker='.', color='blue', s=mksize)
plt.scatter(pointsh1[:, 0], pointsh1[:, 1] - pointsh1[:, 0], label='$H_1$', marker='.', color='red', s=mksize)
plt.scatter(pointsh2[:, 0], pointsh2[:, 1] - pointsh2[:, 0], label='$H_2$', marker='.', color='green', s=mksize)
plt.xlabel('Birth', fontsize=ft_size)
plt.ylabel('Persistence', fontsize=ft_size)
plt.xticks(fontsize=ft_size)
plt.yticks(fontsize=ft_size)

h0_line = plt.Line2D([0], [0], marker='.', color='blue', markersize=markersize_, linestyle='None', label='$H_0$')
h1_line = plt.Line2D([0], [0], marker='.', color='red', markersize=markersize_, linestyle='None', label='$H_1$')
h2_line = plt.Line2D([0], [0], marker='.', color='green', markersize=markersize_, linestyle='None', label='$H_2$')
plt.legend(handles=[h0_line, h1_line, h2_line], fontsize=ft_size)

all_births = np.concatenate([pointsh0[:, 0], pointsh1[:, 0], pointsh2[:, 0]])
all_persistences = np.concatenate([pointsh0[:, 1] - pointsh0[:, 0], pointsh1[:, 1] - pointsh1[:, 0], pointsh2[:, 1] - pointsh2[:, 0]])

all_births = all_births[np.isfinite(all_births)]
all_persistences = all_persistences[np.isfinite(all_persistences)]

max_range = max(np.max(all_births), np.max(all_persistences))
padding = max_range * 0.05
offset = padding / 2

plt.xlim([0 - offset, max_range + padding - offset])
plt.ylim([0 - offset, max_range + padding - offset])
plt.gca().set_aspect('equal', adjustable='box')

plt.show()
