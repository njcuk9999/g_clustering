#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# CODE NAME HERE

# CODE DESCRIPTION HERE

Created on 2018-01-16 at 14:13

@author: cook



Version 0.0.0
"""
import numpy as np
from astropy.table import Table
from sklearn.cluster import DBSCAN
from sklearn import metrics
import random
import matplotlib.pyplot as plt
import time
from sklearn.decomposition import PCA as sklearnPCA

# =============================================================================
# Define variables
# =============================================================================
# Define paths
WORKSPACE = '/scratch/Projects/Gaia_clustering'
WRITEPATH = WORKSPACE + '/data/Sim/Simulation_simple.fits'

# -----------------------------------------------------------------------------
COLOURS = ['r', 'g', 'b', 'c', 'm', 'orange']
MARKERS = ['o', 's', '*', 'd', 'v', '<', '>', '^', 'h', 'D', 'p', '8']

SUBSET = True


# =============================================================================
# Define functions
# =============================================================================
def get_random_choices(array, num):
    mask = random.choices(range(len(array)), k=num)
    return mask


# =============================================================================
# Start of code
# =============================================================================
# Main code here
if __name__ == "__main__":

    # get the data
    print("Loading data...")
    rawdata = Table.read(WRITEPATH)

    # apply subset to data
    if SUBSET:
        mask = get_random_choices(rawdata, 100000)
    else:
        mask = np.ones(len(rawdata['X']), dtype=bool)
    rawdata = rawdata[mask]

    # construct data matrix
    data = np.array([rawdata['X'], rawdata['Y'], rawdata['Z'],
                     rawdata['U'], rawdata['V'], rawdata['W']]).T
    # data = np.array([rawdata['X'], rawdata['Y'], rawdata['Z']]).T

    # get the true labels and group names
    labels_true = np.array(rawdata['row'])
    groups = np.array(rawdata['group'])


    # -------------------------------------------------------------------------
    # pca
    from matplotlib.mlab import PCA as mlabPCA

    mlab_pca = mlabPCA(data)

    # compute contribution fractions
    fracs = mlab_pca.fracs

    for f_it, frac in enumerate(fracs):
        print('Dim {0}: Percentage power = {1:.2f} %'.format(f_it, frac*100))




# =============================================================================
# Start of code
# =============================================================================
# Main code here
if __name__ == "__main__":
    # ----------------------------------------------------------------------
    # print 'Hello World!'
    print("Hello World!")

# =============================================================================
# End of code
# =============================================================================
