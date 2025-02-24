#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

soft_q6_values = []
hard_q6_values = []

for i in [1, 2]:
    for j in range(10):
        for k in [100, 200, 300, 400, 500, 600, 700, 800, 900]:
            q6_file = f"../data/particle_q6/set_{i}/D_{j}/Y_{k}.npy"
            label_file = f"../data/p_hop_t/set_{i}/D_{j}/Y_{k}.npy"
            
            if os.path.exists(q6_file) and os.path.exists(label_file):
                q6_values = np.load(q6_file)
                labels = np.load(label_file)

                soft_q6_values.extend(q6_values[labels == 1])
                hard_q6_values.extend(q6_values[labels == 0])

data = pd.DataFrame({
    'q6_value': soft_q6_values + hard_q6_values,
    'label': ['Soft'] * len(soft_q6_values) + ['Hard'] * len(hard_q6_values)
})

plt.figure(figsize=(6, 6), dpi=300)
sns.boxplot(
    x='label', y='q6_value', data=data,
    hue='label',
    dodge=False,
    palette={'Soft': '#ADD8E6', 'Hard': '#006400'},
    linewidth=2.5
)
if plt.gca().get_legend() is not None:
    plt.gca().get_legend().remove()
plt.xlabel('Particle Type', fontsize=20)
plt.ylabel('q6 Value', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1], fontsize=20)

plt.tight_layout()
plt.savefig('softhard.png')
plt.show()
