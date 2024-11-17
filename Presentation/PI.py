#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from ripser import Rips
from ripser import ripser
from sklearn.base import TransformerMixin
import numpy as np
import collections
from scipy.stats import norm
import matplotlib.pyplot as plt
collections.Iterable = collections.abc.Iterable

pixel_m = 10
pixel_n = 10
r_max = 3
sigma = 0.1

class PI(TransformerMixin):

    def __init__(
        self,
        pixels=(20, 20),
        spread=None,
        specs=None,
        kernel_type="gaussian",
        weighting_type="linear",
        verbose=True,
    ):
        self.specs = specs
        self.kernel_type = kernel_type
        self.weighting_type = weighting_type
        self.spread = spread
        self.nx, self.ny = pixels

        if verbose:
            print(
                'PI(pixels={}, spread={}, specs={}, kernel_type="{}", weighting_type="{}")'.format(
                    pixels, spread, specs, kernel_type, weighting_type
                )
            )

    def transform(self, diagrams):
        if len(diagrams) == 0:
            return np.zeros((self.nx, self.ny))

        singular = not isinstance(diagrams[0][0], collections.Iterable)

        if singular:
            diagrams = [diagrams]

        dgs = [np.copy(diagram) for diagram in diagrams]
        landscapes = [PI.to_landscape(dg) for dg in dgs]

        if not self.specs:
            self.specs = {
                "maxBD": np.max([np.max(np.vstack((landscape, np.zeros((1, 2))))) 
                                 for landscape in landscapes] + [0]),
                "minBD": np.min([np.min(np.vstack((landscape, np.zeros((1, 2))))) 
                                 for landscape in landscapes] + [0]),
            }
        imgs = [self._transform(dgm) for dgm in landscapes]

        if singular:
            imgs = imgs[0]

        return imgs

    def _transform(self, landscape):
        maxBD = self.specs["maxBD"]
        minBD = min(self.specs["minBD"], 0)
        dx = maxBD / (self.ny)
        xs_lower = np.linspace(minBD, maxBD, self.nx)
        xs_upper = np.linspace(minBD, maxBD, self.nx) + dx
        ys_lower = np.linspace(0, maxBD, self.ny)
        ys_upper = np.linspace(0, maxBD, self.ny) + dx
        weighting = self.weighting(landscape)
        img = np.zeros((self.nx, self.ny))

        spread = self.spread if self.spread else dx

        for point in landscape:
            x_smooth = norm.cdf(xs_upper, point[0], spread) - norm.cdf(
                xs_lower, point[0], spread
            )
            y_smooth = norm.cdf(ys_upper, point[1], spread) - norm.cdf(
                ys_lower, point[1], spread
            )
            img += np.outer(x_smooth, y_smooth) * weighting(point)

        img = img.T[::-1]
        return img

    def weighting(self, landscape=None):
        if landscape is not None and len(landscape) > 0:
            maxy = np.max(landscape[:, 1])
        else:
            maxy = 1

        def linear(interval):
            d = interval[1]
            return (1 / maxy) * d if landscape is not None else d

        return linear

    @staticmethod
    def to_landscape(diagram):
        diagram[:, 1] -= diagram[:, 0]
        return diagram

    def show(self, imgs, ax=None):
        ax = ax or plt.gca()
        ax.set_facecolor('white')

        if type(imgs) is not list:
            imgs = [imgs]
        for i, img in enumerate(imgs):
            img_normalized = (img - np.min(img)) / (np.max(img) - np.min(img))
            img_inverted = 1 - img_normalized
            ax.imshow(img_inverted, cmap='gray', alpha=0.5, vmin=0, vmax=1)

        ax.axis("off")


def dist_mat(t):
    element = np.loadtxt(t, dtype=str, usecols=(0,), skiprows=2)
    x = np.loadtxt(t, dtype=float, usecols=(1), skiprows=2)
    y = np.loadtxt(t, dtype=float, usecols=(2), skiprows=2)
    z = np.loadtxt(t, dtype=float, usecols=(3), skiprows=2)
    Distance = np.zeros(shape=(len(x), len(x)))
    for i in range(0, len(x)):
        for j in range(0, len(x)):
            Distance[i][j] = np.sqrt(((x[i] - x[j]) ** 2) + ((y[i] - y[j]) ** 2) + ((z[i] - z[j]) ** 2))
    return [Distance, element]

