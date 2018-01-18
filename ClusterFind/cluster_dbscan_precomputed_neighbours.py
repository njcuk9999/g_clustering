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
from sklearn.neighbors import NearestNeighbors

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
SUBSETSIZE = 500000

DIMNAMES = ['X [pc]', 'Y [pc]', 'Z [pc]',
            'U [mas/yr]', 'V [mas/yr]', 'W [mas/yr]']


# =============================================================================
# Define functions
# =============================================================================
def get_random_choices(array_length, num):
    mask = random.choices(range(array_length), k=num)
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


def plot_data(data, limits=None):
    # get dimensions fitted
    Ndim = data.shape[1]
    # get ranges for graph plotting
    range1 = range(Ndim-1)
    range2 = range(1, Ndim)
    # get optimal grid
    nrows, ncols, pos = optimal_grid(len(range1))
    # set up figure
    fig, frames = plt.subplots(nrows=nrows, ncols=ncols)

    # loop around dimensions (graph positions)
    for it in range(len(range1)):
        # get positions of dimensions in data
        r1, r2 = range1[it], range2[it]
        frame = frames[pos[it][0]][pos[it][1]]
        # plot points
        frame.plot(data[:, r1], data[:, r2], markersize=2,
                   marker='x', alpha=0.1,
                   zorder=1, color='k', linestyle='none')
        # limits
        if limits is not None:
            frame.set(xlim=limits[it][:2], ylim=limits[it][2:])
        # labels
        frame.set(xlabel='{0}'.format(DIMNAMES[r1]),
                  ylabel='{0}'.format(DIMNAMES[r2]))

    # title
    plt.suptitle('Data before clustering')

    # deal with blank frames
    for it in range(len(range1), nrows * ncols):
        frame = frames[pos[it][0]][pos[it][1]]
        frame.axis('off')


def plot_dims(data, labels, n_clusters, kind='out', setlimits=None):

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
    # get optimal grid
    nrows, ncols, pos = optimal_grid(len(range1))
    # set up figure
    fig, frames = plt.subplots(nrows=nrows, ncols=ncols)
    # loop around dimensions (graph positions)
    limits = []
    for it in range(len(range1)):
        # get positions of dimensions in data
        r1, r2 = range1[it], range2[it]
        frame = frames[pos[it][0]][pos[it][1]]
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
                frame.plot(xy[:, r1], xy[:, r2], markersize=2,
                           marker=markers[k_it], alpha=alpha,
                           zorder=zorder, color=colours[k_it], linestyle='none')
                stats = find_min_max(xy[:, r1], xy[:, r2], *stats)
            else:
                frame.plot(xy[:, r1], xy[:, r2], markersize=2,
                           marker='x', alpha=alpha,
                           zorder=zorder, color='k', linestyle='none')
        # set labels
        frame.set(xlabel='{0}'.format(DIMNAMES[r1]),
                  ylabel='{0}'.format(DIMNAMES[r2]))
        # set limits
        if setlimits is None:
            frame.set(xlim=stats[:2], ylim=stats[2:])
            limits.append(stats)
        else:
            frame.set(xlim=setlimits[it][:2], ylim=setlimits[it][2:])
            limits.append(setlimits[it])

    # deal with blank frames
    for it in range(len(range1), nrows * ncols):
        frame = frames[pos[it][0]][pos[it][1]]
        frame.axis('off')


    if kind == 'in':
        plt.suptitle('Simulated number of clusters: {0}'.format(n_clusters))
    else:
        plt.suptitle('Estimated number of clusters: {0}'.format(n_clusters))

    return limits


def find_min_max(x, y, xmin, xmax, ymin, ymax, zoomout=0.10):
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


        printlog('\t Group: {0}  (Total = {1})'.format(ugroup, in_num))
        for key in comp:

            if key == -1:
                ll = 'NOISE (G=-1)'
            elif key in newlabelgroup:
                ll = '{0} (G={1})'.format(newlabelgroup[key], key)
            else:
                ll = 'NEW (G={0})'.format(key)

            printlog('\t\tlabel={0}   number found={1}'.format(ll, comp[key]))

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


def printlog(message):
    message = message.split('\n')
    for mess in message:
        unix_time = time.time()
        human_time = time.strftime('%H:%M:%S', time.localtime(unix_time))
        dsec = int((unix_time - int(unix_time)) * 100)
        print('{0}.{1:02d} | {2}'.format(human_time, dsec, mess))


# =============================================================================
# Start of code
# =============================================================================
# Main code here
if __name__ == "__main__":

    # get the data
    printlog("Loading data...")
    rawdata = Table.read(WRITEPATH)

    # apply subset to data
    if SUBSET:
        mask = get_random_choices(len(rawdata), SUBSETSIZE)
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

    # get nearest neighbours
    printlog('Work out nearest neighbours...')
    start = time.time()
    neigh = NearestNeighbors(radius=20, metric='euclidean')
    neigh.fit(data)
    neighbours = neigh.radius_neighbors_graph(data, mode='distance')
    end = time.time()
    printlog('\t Time taken = {0} s'.format(end - start))

    # ----------------------------------------------------------------------
    # DBscan example from :
    #      scikit-learn.org/stable/modules/clustering.html#dbscan
    #      http://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan
    #          .html#sphx-glr-auto-examples-cluster-plot-dbscan-py
    printlog("Calculating clustering using 'DBSCAN'...")
    start = time.time()

    sargs = dict(eps=10, min_samples=50, metric='precomputed')
    db = DBSCAN(**sargs).fit(neighbours)
    end = time.time()
    # get mask and labels
    labels = db.labels_
    # report timing
    printlog('\t Time taken = {0} s'.format(end - start))

    # ----------------------------------------------------------------------
    # stats
    # Number of clusters in labels, ignoring noise if present.
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_clusters_true = len(set(labels_true))  - (1 if -1 in labels else 0)

    printlog('\t Estimated number of clusters: {0}'.format(n_clusters))
    #print stats
    args = [labels_true, labels]
    pargs = [metrics.homogeneity_score(*args),
             metrics.completeness_score(*args),
             metrics.v_measure_score(*args),
             metrics.adjusted_rand_score(*args),
             metrics.adjusted_mutual_info_score(*args)]
    printlog("\t Homogeneity: {0:.3f}\n\t Completeness: {1:.3f}"
             "\n\t V-measure: {2:.3f}\n\t Adjusted Rand Index: {3:.3f}"
             "\n\t Adjusted Mutual Information: {4:.3f}".format(*pargs))

    # ----------------------------------------------------------------------
    # comparing results
    printlog('Comparing results...')
    compare_results(groups, labels_true, labels)

    # ----------------------------------------------------------------------
    # Plot result
    printlog('Plotting graph...')

    # dont plot all results
    mask = get_random_choices(len(data), 100000)

    limits = plot_dims(data[mask], labels[mask], n_clusters, kind='out')

    limits = plot_dims(data[mask], labels_true[mask], n_clusters_true,
                       kind='in', setlimits=limits)

    plot_data(data[mask], limits=limits)

    plt.show()
    plt.close()


# =============================================================================
# End of code
# =============================================================================
