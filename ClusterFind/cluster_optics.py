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
from pyclustering.cluster import cluster_visualizer
from pyclustering.cluster.optics import optics, ordering_analyser, ordering_visualizer

from pyclustering.utils import read_sample, timedcall

from pyclustering.samples.definitions import SIMPLE_SAMPLES, FCPS_SAMPLES


# Note may require
#           conda update libgcc

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
        mask = get_random_choices(rawdata, 10000)
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

    # ----------------------------------------------------------------------
    # optics example from :
    #      https://raw.githubusercontent.com/annoviko/pyclustering/master/
    #          pyclustering/cluster/examples/optics_examples.py
    print("Calculating clustering using 'OPTICS'...")

    optics_instance = optics(datalist, eps=1, minpts=3, ccore=True)
    (ticks, _) = timedcall(optics_instance.process)

    print("\t\tExecution time: ", ticks, "\n")

    # get clusters and noise
    clusters = optics_instance.get_clusters()
    noise = optics_instance.get_noise()



# =============================================================================
# End of code
# =============================================================================
