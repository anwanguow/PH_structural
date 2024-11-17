#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from ripser import ripser
import numpy as np
import matplotlib.pyplot as plt

set_ = 1
traj_ = 5
frame_ = 900

mksize = 80
ft_size = 30

file = '../../data/distance_matrix/set_' + str(set_) + '/D_' + str(traj_-1) + '/D_' + str(frame_) + '.npy'
D = np.load(file, allow_pickle=True)

a = ripser(D, distance_matrix=True, maxdim=2)

pointsh1 = a['dgms'][1]
pointsh2 = a['dgms'][2]

plt.figure(figsize=(6, 6), dpi=300)
plt.scatter(pointsh1[:, 0], pointsh1[:, 1] - pointsh1[:, 0], marker='.', color='red', s=mksize)
plt.scatter(pointsh2[:, 0], pointsh2[:, 1] - pointsh2[:, 0], marker='.', color='green', s=mksize)
plt.xlabel('Birth', fontsize=ft_size)
plt.ylabel('Persistence', fontsize=ft_size)
plt.xticks([1.0, 1.2, 1.4], fontsize=ft_size)
plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5], fontsize=ft_size)

plt.xlim([0.9, 1.5])
plt.ylim([0, 0.6])
plt.gca().set_aspect('equal', adjustable='box')
plt.show()
