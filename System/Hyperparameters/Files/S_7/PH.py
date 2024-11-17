# -*- coding: utf-8 -*-

from sklearn.base import TransformerMixin
import numpy as np
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
import matplotlib.pyplot as plt

try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable

class PersImage(TransformerMixin):

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
                'PersImage(pixels={}, spread={}, specs={}, kernel_type="{}", weighting_type="{}")'.format(
                    pixels, spread, specs, kernel_type, weighting_type
                )
            )

    def transform(self, diagrams):
        if len(diagrams) == 0:
            return np.zeros((self.nx, self.ny))
        try:
            singular = not isinstance(diagrams[0][0], Iterable)
        except IndexError:
            singular = False
        if singular:
            diagrams = [diagrams]

        dgs = [np.copy(diagram) for diagram in diagrams]
        landscapes = [PersImage.to_landscape(dg) for dg in dgs]

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
        if np.size(landscape,1) == 2:
            
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
        else:
            spread = self.spread if self.spread else dx
            for point in landscape:
                x_smooth = norm.cdf(xs_upper, point[0], point[2]*spread) - norm.cdf(
                    xs_lower, point[0], point[2]*spread
                )
                y_smooth = norm.cdf(ys_upper, point[1], point[2]*spread) - norm.cdf(
                    ys_lower, point[1], point[2]*spread
                )
                img += np.outer(x_smooth, y_smooth) * weighting(point)
            img = img.T[::-1]
            return img

    def weighting(self, landscape=None):
        if landscape is not None:
            if len(landscape) > 0:
                maxy = np.max(landscape[:, 1])
            else: 
                maxy = 1
        def linear(interval):
            # linear function of y such that f(0) = 0 and f(max(y)) = 1
            d = interval[1]
            return (1 / maxy) * d if landscape is not None else d
        def pw_linear(interval):
            t = interval[1]
            b = maxy / self.ny
            if t <= 0:
                return 0
            if 0 < t < b:
                return t / b
            if b <= t:
                return 1

        return linear

    def kernel(self, spread=1):
        def gaussian(data, pixel):
            return mvn.pdf(data, mean=pixel, cov=spread)
        return gaussian

    @staticmethod
    def to_landscape(diagram):
        diagram[:, 1] -= diagram[:, 0]
        return diagram

    def show(self, imgs, ax=None):
        ax = ax or plt.gca()
        if type(imgs) is not list:
            imgs = [imgs]
        for i, img in enumerate(imgs):
            ax.imshow(img, cmap=plt.get_cmap("plasma"))
            ax.axis("off")

def PI_vector(barcode, pixelx=20, pixely=20, myspread=0.1, myspecs={"maxBD": 15, "minBD":-0.1}, showplot=False):
    Totalmatrix=barcode
    pim = PersImage(pixels=[pixelx,pixely], spread=myspread, specs=myspecs, verbose=False)
    imgs = pim.transform(Totalmatrix)
   
    if showplot == True:
        pim.show(imgs)
        plt.show()
    return np.array(imgs.flatten())

def PI_matrix(barcode, pixelx=20, pixely=20, myspread=0.1, myspecs={"maxBD": 15, "minBD":-0.1}, showplot=False):
    Totalmatrix=barcode
    pim = PersImage(pixels=[pixelx,pixely], spread=myspread, specs=myspecs, verbose=False)
    imgs = pim.transform(Totalmatrix)
   
    if showplot == True:
        pim.show(imgs)
        plt.show()
    return np.array(imgs)

from ripser import Rips
rips = Rips(maxdim=2)

def PH_barcode(D_matrix):
    D = D_matrix;
    diagrams = rips.fit_transform(D, distance_matrix=True)
    h0matrix=diagrams[0][0:-1,:]
    h1matrix=diagrams[1]
    h2matrix=diagrams[2]
    Totalmatrix=np.vstack((h0matrix,h1matrix,h2matrix))
    return Totalmatrix

def PH_barcode_h0(D_matrix):
    D = D_matrix;
    diagrams = rips.fit_transform(D, distance_matrix=True)
    h0matrix=diagrams[0][0:-1,:]
    return h0matrix

def PH_barcode_h1(D_matrix):
    D = D_matrix;
    diagrams = rips.fit_transform(D, distance_matrix=True)
    h1matrix=diagrams[1]
    return h1matrix

def PH_barcode_h2(D_matrix):
    D = D_matrix;
    diagrams = rips.fit_transform(D, distance_matrix=True)
    h2matrix=diagrams[2]
    return h2matrix

def PH_barcode_h0h1(D_matrix):
    D = D_matrix;
    diagrams = rips.fit_transform(D, distance_matrix=True)
    h0matrix=diagrams[0][0:-1,:]
    h1matrix=diagrams[1]
    Totalmatrix=np.vstack((h0matrix,h1matrix))
    return Totalmatrix

def PH_barcode_h0h2(D_matrix):
    D = D_matrix;
    diagrams = rips.fit_transform(D, distance_matrix=True)
    h0matrix=diagrams[0][0:-1,:]
    h2matrix=diagrams[2]
    Totalmatrix=np.vstack((h0matrix,h2matrix))
    return Totalmatrix

def PH_barcode_h1h2(D_matrix):
    D = D_matrix;
    diagrams = rips.fit_transform(D, distance_matrix=True)
    h1matrix=diagrams[1]
    h2matrix=diagrams[2]
    Totalmatrix=np.vstack((h1matrix,h2matrix))
    return Totalmatrix

def PH_barcode_h0h1h2(D_matrix):
    D = D_matrix;
    diagrams = rips.fit_transform(D, distance_matrix=True)
    h0matrix=diagrams[0][0:-1,:]
    h1matrix=diagrams[1]
    h2matrix=diagrams[2]
    Totalmatrix=np.vstack((h0matrix,h1matrix,h2matrix))
    return Totalmatrix


