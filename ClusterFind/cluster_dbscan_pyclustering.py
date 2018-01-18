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
import random
import matplotlib.pyplot as plt

from pyclustering.cluster import cluster_visualizer
from pyclustering.cluster.dbscan import dbscan

from pyclustering.utils import read_sample
from pyclustering.utils import timedcall

from pyclustering.samples.definitions import SIMPLE_SAMPLES, FCPS_SAMPLES


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
SUBSETSIZE = 100000

DIMNAMES = ['X [pc]', 'Y [pc]', 'Z [pc]',
            'U [mas/yr]', 'V [mas/yr]', 'W [mas/yr]']


# =============================================================================
# Define functions
# =============================================================================
def get_random_choices(array, num):
    mask = random.choices(range(len(array)), k=num)
    return mask


def optimal_grid(num):

    # get maximum shape
    shape = int(np.ceil(np.sqrt(num)))
    # get number of rows and columns based on maximum shape
    if shape ** 2 == num:
        nrows = shape
        ncols = shape
    else:
        nrows = int(np.ceil(num / shape))
        ncols = int(np.ceil(num / nrows))
    # get position of figures
    pos = []
    for i in range(nrows):
        for j in range(ncols):
            pos.append([i, j])
    # return nrows, ncols and positions
    return nrows, ncols, pos


def plot_selection(data, clusters, noise):



    plt.scatter(data[:, 0], data[:, 1], marker='.')

    clustermask = get_mask(clusters)
    noisemask = np.array(noise)

    plt.scatter(data[:, 0][noisemask], data[:, 1][noisemask],
                color='k', marker='.', s=1)

    plt.scatter(data[:, 0][clustermask], data[:, 1][clustermask],
                color='r', marker='x')

    plt.show()
    plt.close()


def get_mask(ll):

    mask = []
    for l in range(len(ll)):
        mask = np.append(mask, ll[l])
    mask = np.array(mask, dtype=int)
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
        mask = get_random_choices(rawdata, SUBSETSIZE)
    else:
        mask = np.ones(len(rawdata['X']), dtype=bool)
    rawdata = rawdata[mask]

    # construct data matrix
    data = np.array([rawdata['X'], rawdata['Y'], rawdata['Z'],
                     rawdata['U'], rawdata['V'], rawdata['W']]).T
    # data = np.array([rawdata['X'], rawdata['Y'], rawdata['Z']]).T

    datalist = []
    for row in range(data.shape[0]):
        datalist.append(list(data[row]))


    # get the true labels and group names
    labels_true = np.array(rawdata['row'])
    groups = np.array(rawdata['group'])

    # convert data to 32 bit
    data = np.array(data, dtype=np.float32)

    # ----------------------------------------------------------------------
    # DBscan example from :
    #      scikit-learn.org/stable/modules/clustering.html#dbscan
    #      http://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan
    #          .html#sphx-glr-auto-examples-cluster-plot-dbscan-py
    print("Calculating clustering using 'DBSCAN (pyclustering)'...")

    dbscan_instance = dbscan(data=datalist, eps=10, neighbors=10, ccore=True)
    (ticks, _) = timedcall(dbscan_instance.process)

    print("\t\tExecution time: ", ticks, "\n")

    clusters = dbscan_instance.get_clusters()
    noise = dbscan_instance.get_noise()

    plot_selection(data, clusters, noise)

# =============================================================================
# End of code
# =============================================================================
