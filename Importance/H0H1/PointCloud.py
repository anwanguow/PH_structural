#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

def get_neighbor(D, i, r):
    D_i = D[i]
    indices = np.where(D_i <= r)[0]
    neighbor = D[np.ix_(indices, indices)]
    return neighbor, indices
