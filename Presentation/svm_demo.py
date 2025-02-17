#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

np.random.seed(0)

X = np.r_[np.random.randn(30, 2) * 4.5 - [10, 10], np.random.randn(30, 2) * 4.5 + [10, 10]]
Y = np.array([0] * 30 + [1] * 30)

clf = svm.SVC(kernel='linear')
clf.fit(X, Y)

w = clf.coef_[0]
b = clf.intercept_[0]
a = -w[0] / w[1]
xx = np.linspace(-20, 20)
yy = a * xx - (b / w[1])

margin_down = a * xx + (clf.support_vectors_[0, 1] - a * clf.support_vectors_[0, 0])
margin_up = a * xx + (clf.support_vectors_[-1, 1] - a * clf.support_vectors_[-1, 0])

positive_points = X[Y == 1]
max_x = np.max(positive_points[:, 0])
rightmost_points = positive_points[positive_points[:, 0] == max_x]
rightmost_lowest_positive = rightmost_points[np.argmin(rightmost_points[:, 1])]

proj_x = (rightmost_lowest_positive[0] - w[0] * (np.dot(w, rightmost_lowest_positive) + b) / np.linalg.norm(w)**2)
proj_y = (rightmost_lowest_positive[1] - w[1] * (np.dot(w, rightmost_lowest_positive) + b) / np.linalg.norm(w)**2)

plt.figure(figsize=(5, 5), dpi=300)
plt.plot(xx, yy, 'k-', label='Decision Hyperplane')
plt.plot(xx, margin_down, 'k--', label='Margin')
plt.plot(xx, margin_up, 'k--')

plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
            s=120, facecolors='none', edgecolors='k', label='Support Vectors')
plt.scatter(X[Y == 0][:, 0], X[Y == 0][:, 1], c='blue', edgecolors='k', label='Negative Class (Hard)')
plt.scatter(X[Y == 1][:, 0], X[Y == 1][:, 1], c='red', edgecolors='k', label='Positive Class (Soft)')

plt.annotate('', xy=(proj_x, proj_y), xytext=(rightmost_lowest_positive[0], rightmost_lowest_positive[1]),
             arrowprops=dict(facecolor='darkgoldenrod', edgecolor='darkgoldenrod', linewidth=2, arrowstyle='->'),
             label='Softness')

plt.axis('equal')
plt.xlim(-25, 25)
plt.ylim(-25, 25)
plt.xticks([])
plt.yticks([])

plt.savefig("svm_demo.png", dpi=300, bbox_inches='tight')
plt.show()

fig_legend = plt.figure(figsize=(3, 3), dpi=300)
ax_legend = fig_legend.add_subplot(111)
ax_legend.axis('off')

scatter1 = ax_legend.scatter([], [], c='blue', edgecolors='k', label='Negative \nClass (Hard)')
scatter2 = ax_legend.scatter([], [], c='red', edgecolors='k', label='Positive \nClass (Soft)')
scatter3 = ax_legend.scatter([], [], facecolors='none', edgecolors='k', s=120, label='Support \nVectors')
line1, = ax_legend.plot([], [], 'k-', label='Decision \nHyperplane')
line2, = ax_legend.plot([], [], 'k--', label='Support \nVector \nHyperplane')
arrow_legend, = ax_legend.plot([], [], color='darkgoldenrod', linewidth=2, marker='>', markersize=8, label='Softness')

ax_legend.legend(loc='center', frameon=False)

plt.show()


