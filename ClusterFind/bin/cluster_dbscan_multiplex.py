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
import DBSCAN_multiplex as DB
from sklearn import metrics
import random
import matplotlib.pyplot as plt
import time


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





def plot_dims(data, labels, n_clusters, kind='out'):

    # get unique labels
    unique_labels = np.unique(labels)
    # get colour marker combinations
    colours = np.tile(COLOURS, len(MARKERS))
    markers = np.repeat(MARKERS, len(COLOURS))
    # make sure we are not repeating
    while len(unique_labels) > len(markers):
        colours = np.repeat(colours, 2)
        markers = np.repeat(markers, 2)


    # get dimensions fitted
    Ndim = data.shape[1]
    # get ranges for graph plotting
    range1 = range(Ndim-1)
    range2 = range(1, Ndim)
    # get best shape
    shape = int(np.ceil(np.sqrt(Ndim - 1)))
    # set up figure
    fig, frames = plt.subplots(nrows=shape, ncols=shape)

    # loop around dimensions (graph positions)
    for pos in range(shape**2):
        # get position in plot
        i, j = pos//shape, pos % shape
        frame = frames[i][j]
        # deal with blank plots
        if pos >= len(range1):
            frames[i][j].axis('off')
            continue
        # get positions of dimensions in data
        r1, r2 = range1[pos], range2[pos]

        stats = [0.0, 0.0, 0.0, 0.0]
        # loop around groups
        for k_it in unique_labels:
            # get members for this group
            class_member_mask = (labels == k_it)
            # if noise set the colour to black
            if k_it == -1:
                alpha = 0.1
                zorder = 1
            else:
                alpha = 1.0
                zorder = 2

            # plot points in the core sample
            xy = data[class_member_mask]
            if k_it != -1:
                frame.plot(xy[:, r1], xy[:, r2], markersize=5,
                           marker=markers[k_it], alpha=alpha,
                           zorder=zorder, color=colours[k_it], linestyle='none')
                stats = find_min_max(xy[:, r1], xy[:, r2], *stats)
            else:
                frame.plot(xy[:, r1], xy[:, r2], markersize=5,
                           marker='+', alpha=alpha,
                           zorder=zorder, color='k', linestyle='none')

            frame.set(xlabel='Dim={0}'.format(r1),
                      ylabel='Dim={0}'.format(r2))

        frame.set(xlim=stats[:2], ylim=stats[2:])

    if kind == 'in':
        plt.suptitle('Simulated number of clusters: {0}'.format(n_clusters))
    else:
        plt.suptitle('Estimated number of clusters: {0}'.format(n_clusters))


def find_min_max(x, y, xmin, xmax, ymin, ymax, zoomout=0.05):
    """
    Takes arrays of x and y and tests limits against previously defined limits
    if limits are exceeded limits are changed with a zoom out factor

    :param x: array, x values
    :param y: array, yvalues
    :param xmin: float, old xmin value to be tested
    :param xmax: float, old xmax value to be tested
    :param ymin: float, old ymin value to be tested
    :param ymax: float, old ymax value to be tested
    :param zoomout: float, the fraction zoomout factor i.e. 0.05 = 5% zoomout
                    to zoom in make number negative, for no zoomout put it to
                    zero
    :return:
    """
    if len(x) != 0:
        newxmin, newxmax = np.min(x), np.max(x)
        diffx = newxmax - newxmin
        if newxmin < xmin:
            xmin = newxmin - zoomout * diffx
        if newxmax > xmax:
            xmax = newxmax + zoomout * diffx

    if len(y) != 0:
        newymin, newymax = np.min(y), np.max(y)
        diffy = newymax - newymin
        if newymin < ymin:
            ymin = newymin - zoomout * diffy
        if newymax > ymax:
            ymax = newymax + zoomout * diffy
    return xmin, xmax, ymin, ymax


def compare_results(groups, labels_true, labels):

    ugroups = np.unique(groups)

    newlabelgroup = dict()

    for ugroup in ugroups:

        # find the key for this ugroup
        mask = groups == ugroup

        in_num = np.sum(mask)
        # make sure we only have one label per group (we should)
        glabels = labels_true[mask]
        if len(np.unique(glabels)) > 1:
            raise ValueError('Group {0} has more than one key!'.format(ugroup))
        else:
            ulabel = glabels[0]
        # get label mask
        mask = labels_true == ulabel
        # count the number of labels in group
        comp = counter(labels[mask])


        print('\n\t Group: {0}  (Total = {1})'.format(ugroup, in_num))
        for key in comp:

            if key == -1:
                ll = 'NOISE (G=-1)'
            elif key in newlabelgroup:
                ll = '{0} (G={1})'.format(newlabelgroup[key], key)
            else:
                ll = 'NEW (G={0})'.format(key)

            print('\t\tlabel={0}   number found={1}'.format(ll, comp[key]))

            if key == -1:
                newlabelgroup[key] = 'NOISE'
            elif key not in newlabelgroup:
                newlabelgroup[key] = ugroup


def counter(array):
    ddict = dict()
    for a in array:
        if a not in ddict:
            ddict[a] = 1
        else:
            ddict[a] += 1

    # reverse sort by values
    sort = np.argsort(list(ddict.values()))[::-1]

    keys = np.array(list(ddict.keys()))[sort]
    values = np.array(list(ddict.values()))[sort]

    ddict2 = dict(zip(keys, values))

    return ddict2


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

    # convert data to 32 bit
    data = np.array(data, dtype=np.float32)

    # ----------------------------------------------------------------------
    # DBscan example from :
    #      scikit-learn.org/stable/modules/clustering.html#dbscan
    #      http://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan
    #          .html#sphx-glr-auto-examples-cluster-plot-dbscan-py
    print("Calculating clustering using 'DBSCAN'...")
    start = time.time()

    sargs = dict(eps=100, min_samples=10)
    _, labels = DB.DBSCAN(data, minPts=10)
    end = time.time()
    # report timing
    print('\n\t Time taken = {0} s'.format(end - start))

    # ----------------------------------------------------------------------
    # stats
    # Number of clusters in labels, ignoring noise if present.
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_clusters_true = len(set(labels_true))  - (1 if -1 in labels else 0)

    print('\n\t Estimated number of clusters: {0}'.format(n_clusters))
    #print stats
    args = [labels_true, labels]
    pargs = [metrics.homogeneity_score(*args),
             metrics.completeness_score(*args),
             metrics.v_measure_score(*args),
             metrics.adjusted_rand_score(*args),
             metrics.adjusted_mutual_info_score(*args)]
    print(("\n\t Homogeneity: {0:.3f}\n\t Completeness: {1:.3f}"
           "\n\t V-measure: {2:.3f}\n\t Adjusted Rand Index: {3:.3f}"
           "\n\t Adjusted Mutual Information: {4:.3f}").format(*pargs))

    # ----------------------------------------------------------------------
    # comparing results
    compare_results(groups, labels_true, labels)

    # ----------------------------------------------------------------------
    # Plot result
    print('Plotting graph...')
    plot_dims(data, labels, n_clusters, kind='out')

    plot_dims(data, labels_true, n_clusters_true, kind='in')

    plt.show()
    plt.close()


# =============================================================================
# End of code
# =============================================================================
