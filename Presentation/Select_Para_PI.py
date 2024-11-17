#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from PH import PI_matrix
from PH import PH_barcode

file = '../data/distance_matrix/set_2/D_3/D_100.npy'
D = np.load(file, allow_pickle=True)

myspread = 0.002

pixelx = 40
pixely = pixelx
max_bd = 1.5

barcode = PH_barcode(D)
PI = PI_matrix(barcode, pixelx=pixelx, pixely=pixely, myspread=myspread, myspecs={"maxBD": max_bd, "minBD":-0.1}, showplot=True)

