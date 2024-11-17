import numpy as np
import collections
import matplotlib.pyplot as plt
from ripser import Rips, ripser
from PointCloud import get_neighbor
import random

collections.Iterable = collections.abc.Iterable

def compute_betti_numbers_at_radius(dgms, radius):
    betti_0 = np.sum((dgms[0][:, 0] <= radius) & (dgms[0][:, 1] > radius))
    betti_1 = np.sum((dgms[1][:, 0] <= radius) & (dgms[1][:, 1] > radius))
    betti_2 = np.sum((dgms[2][:, 0] <= radius) & (dgms[2][:, 1] > radius))
    return betti_0, betti_1, betti_2

def get_persistent_homology_data(set_, traj_, frame_, particle_, r_):
    file = f'../../data/distance_matrix/set_{set_}/D_{traj_}/D_{frame_}.npy'
    D = np.load(file, allow_pickle=True)
    neighbor, indices = get_neighbor(D, particle_-1, r_)
    label_file = f'../../data/p_hop_t/set_{set_}/D_{traj_}/Y_{frame_}.npy'
    label_data = np.load(label_file, allow_pickle=True)
    label = label_data[particle_-1]

    rips = Rips(maxdim=2)
    a = ripser(neighbor, distance_matrix=True, maxdim=2)
    
    pointsh0 = a['dgms'][0]
    pointsh1 = a['dgms'][1]
    pointsh2 = a['dgms'][2]

    return label, pointsh0, pointsh1, pointsh2

r_ = 2.5

labels_0 = 0
labels_1 = 0

labels_0_data = []
labels_1_data = []

selected_ = 100

while labels_0 < selected_ or labels_1 < selected_:
    set_ = random.randint(1, 2)
    traj_ = random.randint(0, 9)
    frame_ = random.choice(range(100, 1000, 100))
    particle_ = random.randint(0, 863)
    
    label, pointsh0, pointsh1, pointsh2 = get_persistent_homology_data(set_, traj_, frame_, particle_, r_)
    
    if label == 0 and labels_0 < selected_:
        labels_0_data.append((pointsh0, pointsh1, pointsh2))
        labels_0 += 1
    elif label == 1 and labels_1 < selected_:
        labels_1_data.append((pointsh0, pointsh1, pointsh2))
        labels_1 += 1

plt.figure(figsize=(6, 6), dpi=300)
for i, (pointsh0, pointsh1, pointsh2) in enumerate(labels_0_data):
    plt.scatter(pointsh1[:, 0], pointsh1[:, 1] - pointsh1[:, 0], marker='.', color='red', s=80)
    plt.scatter(pointsh2[:, 0], pointsh2[:, 1] - pointsh2[:, 0], marker='.', color='green', s=80)

plt.xlabel('Birth', fontsize=30)
plt.ylabel('Persistence', fontsize=30)
plt.xticks([1.0,1.2,1.4,1.6], fontsize=30)
plt.yticks([0,0.2,0.4,0.6], fontsize=30)

all_births = np.concatenate([pointsh1[:, 0] for _, pointsh1, _ in labels_0_data])
all_persistences = np.concatenate([pointsh1[:, 1] - pointsh1[:, 0] for _, pointsh1, _ in labels_0_data])

all_births = np.concatenate((all_births, np.concatenate([pointsh2[:, 0] for _, _, pointsh2 in labels_0_data])))
all_persistences = np.concatenate((all_persistences, np.concatenate([pointsh2[:, 1] - pointsh2[:, 0] for _, _, pointsh2 in labels_0_data])))

all_births = all_births[np.isfinite(all_births)]
all_persistences = all_persistences[np.isfinite(all_persistences)]

max_range = max(np.max(all_births), np.max(all_persistences))
padding = max_range * 0.05
offset = padding / 2

plt.xlim([0.9, 1.6])
plt.ylim([0, 0.7])
plt.gca().set_aspect('equal', adjustable='box')


plt.show()

plt.figure(figsize=(6, 6), dpi=300)
for i, (pointsh0, pointsh1, pointsh2) in enumerate(labels_1_data):
    plt.scatter(pointsh1[:, 0], pointsh1[:, 1] - pointsh1[:, 0], marker='.', color='red', s=80)
    plt.scatter(pointsh2[:, 0], pointsh2[:, 1] - pointsh2[:, 0], marker='.', color='green', s=80)


plt.xlabel('Birth', fontsize=30)
plt.ylabel('Persistence', fontsize=30)
plt.xticks([1.0,1.2,1.4,1.6], fontsize=30)
plt.yticks([0,0.2,0.4,0.6], fontsize=30)


all_births = np.concatenate([pointsh1[:, 0] for _, pointsh1, _ in labels_1_data])
all_persistences = np.concatenate([pointsh1[:, 1] - pointsh1[:, 0] for _, pointsh1, _ in labels_1_data])

all_births = np.concatenate((all_births, np.concatenate([pointsh2[:, 0] for _, _, pointsh2 in labels_1_data])))
all_persistences = np.concatenate((all_persistences, np.concatenate([pointsh2[:, 1] - pointsh2[:, 0] for _, _, pointsh2 in labels_1_data])))

all_births = all_births[np.isfinite(all_births)]
all_persistences = all_persistences[np.isfinite(all_persistences)]

max_range = max(np.max(all_births), np.max(all_persistences))
padding = max_range * 0.05
offset = padding / 2

plt.xlim([0.95, 1.6])
plt.ylim([0, 0.65])
plt.gca().set_aspect('equal', adjustable='box')


plt.show()
