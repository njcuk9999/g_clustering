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
from matplotlib.lines import Line2D
import time
from mpl_toolkits.mplot3d import Axes3D


# =============================================================================
# Define variables
# =============================================================================
# Define paths
WORKSPACE = '/scratch/Projects/Gaia_clustering'
WRITEPATH = WORKSPACE + '/data/Sim/Simulation_simple.fits'

# -----------------------------------------------------------------------------
MARKERS = ['o', 'x', '+', '*']

SUBSET = False
SUBSETSIZE = 100000

DIMNAMES = ['X [pc]', 'Y [pc]', 'Z [pc]',
            'U [mas/yr]', 'V [mas/yr]', 'W [mas/yr]']

PLOT_BACKGROUND = True
PLOT_3D_ALL = False
PLOT_3D_MERGED = False
PLOT_2D_ALL = True


# =============================================================================
# Define functions
# =============================================================================

def normalise_data(data):

    skipcols = ['row', 'group']

    newdata = Table()

    for col in data.colnames:
        if col in skipcols:
            newdata[col] = np.array(data[col])
        else:
            maxdata = np.max(data[col])
            newdata[col] = np.array(data[col])/maxdata


    return newdata


def get_random_choices(array_length, num):
    rmask = random.choices(range(array_length), k=num)
    return rmask


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


def plot_data(d):
    # get dimensions fitted
    ndim = d.shape[1]
    # get ranges for graph plotting
    range1 = range(ndim - 1)
    range2 = range(1, ndim)
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
        frame.plot(d[:, r1], d[:, r2], markersize=2,
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


def plot_data3d(d, setlimits=None):
    # set up figure
    plt.figure()
    frame1 = plt.subplot(121, projection='3d')
    frame2 = plt.subplot(122, projection='3d')

    # plot points
    frame1.plot(d[:, 0], d[:, 1], d[:, 2], markersize=2,
                marker='x', alpha=0.1,
                zorder=1, color='k', linestyle='none')
    frame2.plot(d[:, 3], d[:, 4], d[:, 5], markersize=2,
                marker='x', alpha=0.1,
                zorder=1, color='k', linestyle='none')

    # set limits
    if setlimits is not None:
        frame1.set(xlim=setlimits[:2], ylim=setlimits[2:4],
                   zlim=setlimits[4:6])
        frame2.set(xlim=setlimits[6:8], ylim=setlimits[8:10],
                   zlim=setlimits[10:12])

    # set labels
    frame1.set(xlabel='{0}'.format(DIMNAMES[0]),
               ylabel='{0}'.format(DIMNAMES[1]),
               zlabel='{0}'.format(DIMNAMES[2]))
    frame2.set(xlabel='{0}'.format(DIMNAMES[3]),
               ylabel='{0}'.format(DIMNAMES[4]),
               zlabel='{0}'.format(DIMNAMES[5]))

    # title
    plt.suptitle('Data before clustering')


def plot_dims(d, idlabels, nclusters, kind='out', setlimits=None):
    # get unique labels
    unique_labels = np.unique(idlabels)
    # get colour marker combinations
    colors = [plt.cm.jet(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    marks = np.random.choice(MARKERS, size=len(unique_labels))
    # get dimensions fitted
    ndim = d.shape[1]
    # get ranges for graph plotting
    range1 = range(ndim - 1)
    range2 = range(1, ndim)
    # get optimal grid
    nrows, ncols, pos = optimal_grid(len(range1))
    # set up figure
    fig, frames = plt.subplots(nrows=nrows, ncols=ncols)
    # loop around dimensions (graph positions)
    grphlimits = []
    for it in range(len(range1)):
        # get positions of dimensions in data
        r1, r2 = range1[it], range2[it]
        frame = frames[pos[it][0]][pos[it][1]]
        stats = [0.0, 0.0, 0.0, 0.0]
        # loop around groups
        for k_it in unique_labels:
            # get members for this group
            class_member_mask = (idlabels == k_it)
            # if noise set the colour to black
            if k_it == -1:
                alpha = 0.1
                zorder = 1
                if not PLOT_BACKGROUND:
                    continue
            else:
                alpha = 1.0
                zorder = 2
            # plot points in the core sample
            xy = d[class_member_mask]
            if k_it != -1:
                frame.plot(xy[:, r1], xy[:, r2], markersize=2,
                           marker=marks[k_it], alpha=alpha,
                           zorder=zorder, color=tuple(colors[k_it]),
                           linestyle='none')
                stats = find_min_max(xy[:, r1], xy[:, r2], *stats)
            else:
                frame.plot(xy[:, r1], xy[:, r2], markersize=1,
                           marker='.', zorder=zorder, color='k',
                           linestyle='none')
        # set labels
        frame.set(xlabel='{0}'.format(DIMNAMES[r1]),
                  ylabel='{0}'.format(DIMNAMES[r2]))
        # set limits
        if setlimits is None:
            frame.set(xlim=stats[:2], ylim=stats[2:])
            grphlimits.append(stats)
        else:
            frame.set(xlim=setlimits[it][:2], ylim=setlimits[it][2:])
            grphlimits.append(setlimits[it])

    # deal with blank frames
    for it in range(len(range1), nrows * ncols):
        frame = frames[pos[it][0]][pos[it][1]]
        frame.axis('off')

    title = ('\n Number of stars = {0}    Number of stars in groups = {1}'
             '\n Fraction = {2:.4f}')
    targs = [len(d), np.sum(idlabels != -1),
             np.sum(idlabels != -1) / (1.0 * len(d))]
    if kind == 'in':
        suptitle = 'Simulated number of clusters: {0}'.format(nclusters)
        suptitle += title.format(*targs)
    else:
        suptitle = 'Estimated number of clusters: {0}'.format(nclusters)
        suptitle += title.format(*targs)

    plt.suptitle(suptitle)

    return grphlimits


def plot_dims3d(d, idlabels, nclusters, kind='out', setlimits=None,
                names=None):
    # get unique labels
    unique_labels = np.unique(idlabels)
    # get colour marker combinations
    colors = [plt.cm.jet(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    marks = np.random.choice(MARKERS, size=len(unique_labels))

    # set up figure
    plt.figure()
    frame1 = plt.subplot(131, projection='3d')
    frame2 = plt.subplot(132, projection='3d')
    frame3 = plt.subplot(133)

    stats = np.zeros(12)

    # loop around groups
    for it in range(len(unique_labels)):
        # get members for this group
        k_it = unique_labels[it]
        class_member_mask = (idlabels == k_it)
        # if noise set the colour to black
        if k_it == -1:
            alpha = 0.1
            zorder = 1
            if not PLOT_BACKGROUND:
                continue
        else:
            alpha = 1.0
            zorder = 2
        # plot points in the core sample
        xy = d[class_member_mask]
        if k_it != -1:
            frame1.plot(xy[:, 0], xy[:, 1], xy[:, 2], markersize=2,
                        marker=marks[it], alpha=alpha,
                        zorder=zorder, color=tuple(colors[it]),
                        linestyle='none')
            frame2.plot(xy[:, 3], xy[:, 4], xy[:, 5], markersize=2,
                        marker=marks[it], alpha=alpha,
                        zorder=zorder, color=tuple(colors[it]),
                        linestyle='none')

            stats = get_6d_stats(xy, stats)

        else:
            frame1.plot(xy[:, 0], xy[:, 1], xy[:, 2], markersize=1,
                        marker='.', zorder=zorder, color='k',
                        linestyle='none')
            frame2.plot(xy[:, 3], xy[:, 4], xy[:, 5], markersize=1,
                        marker='.', zorder=zorder, color='k',
                        linestyle='none')

    # set labels
    frame1.set(xlabel='{0}'.format(DIMNAMES[0]),
               ylabel='{0}'.format(DIMNAMES[1]),
               zlabel='{0}'.format(DIMNAMES[2]))
    frame2.set(xlabel='{0}'.format(DIMNAMES[3]),
               ylabel='{0}'.format(DIMNAMES[4]),
               zlabel='{0}'.format(DIMNAMES[5]))

    # set limits
    if setlimits is None:
        frame1.set(xlim=stats[:2], ylim=stats[2:4], zlim=stats[4:6])
        frame2.set(xlim=stats[6:8], ylim=stats[8:10], zlim=stats[10:12])
        grphlimits = stats
    else:
        frame1.set(xlim=setlimits[:2], ylim=setlimits[2:4],
                   zlim=setlimits[4:6])
        frame2.set(xlim=setlimits[6:8], ylim=setlimits[8:10],
                   zlim=setlimits[10:12])
        grphlimits = stats

    title = ('\n Number of stars = {0}    Number of stars in groups = {1}'
             '\n Fraction = {2:.4f}')
    targs = [len(d), np.sum(idlabels != -1),
             np.sum(idlabels != -1) / (1.0 * len(d))]
    if kind == 'in':
        suptitle = 'Simulated number of clusters: {0}'.format(nclusters)
        suptitle += title.format(*targs)
    else:
        suptitle = 'Estimated number of clusters: {0}'.format(nclusters)
        suptitle += title.format(*targs)

    plt.suptitle(suptitle)

    # deal with frame 3 (legend)

    handles, idlabels = [], []
    for it in range(len(unique_labels)):
        if names is not None:
            idlabels.append(names[unique_labels[it]])
        else:
            idlabels.append(unique_labels[it])
        handles.append(Line2D([0], [0], color=colors[it], lw=0,
                              marker='o', ms=10))
    frame3.axis('off')
    frame3.legend(handles, idlabels, loc=10)

    return grphlimits


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


def find_min_max_1d(x, xmin, xmax, zoomout=0.10):
    """
    Takes arrays of x and y and tests limits against previously defined limits
    if limits are exceeded limits are changed with a zoom out factor

    :param x: array, x values
    :param xmin: float, old xmin value to be tested
    :param xmax: float, old xmax value to be tested
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
    return xmin, xmax


def get_6d_stats(xy, stats):
    stats[0], stats[1] = find_min_max_1d(xy[:, 0], stats[0], stats[1])
    stats[2], stats[3] = find_min_max_1d(xy[:, 1], stats[2], stats[3])
    stats[4], stats[5] = find_min_max_1d(xy[:, 2], stats[4], stats[5])
    stats[6], stats[7] = find_min_max_1d(xy[:, 3], stats[6], stats[7])
    stats[8], stats[9] = find_min_max_1d(xy[:, 4], stats[8], stats[9])
    stats[10], stats[11] = find_min_max_1d(xy[:, 5], stats[10],
                                           stats[11])
    return stats


def compare_results(allgroups, idlabels_true, idlabels):
    ugroups = np.unique(allgroups)

    newlabelgroup = dict()
    allmerged = []

    for ugroup in ugroups:

        # find the key for this ugroup
        gmask = allgroups == ugroup

        in_num = np.sum(gmask)
        # make sure we only have one label per group (we should)
        glabels = idlabels_true[gmask]
        if len(np.unique(glabels)) > 1:
            raise ValueError('Group {0} has more than one key!'.format(ugroup))
        else:
            ulabel = glabels[0]
        # get label mask
        lmask = idlabels_true == ulabel
        # count the number of labels in group
        comp = counter(idlabels[lmask])

        printlog('\t Group: {0}  (Total = {1})'.format(ugroup, in_num))
        for key in comp:

            if key == -1:
                ll = 'NOISE (G=-1)'
            elif key in newlabelgroup:
                ll = '{0} (G={1})'.format(newlabelgroup[key], key)
                allmerged.append([ugroup, newlabelgroup[key]])
            else:
                ll = 'NEW (G={0})'.format(key)

            printlog('\t\tlabel={0}   number found={1}'.format(ll, comp[key]))

            if key == -1:
                newlabelgroup[key] = 'NOISE'
            elif key not in newlabelgroup:
                newlabelgroup[key] = ugroup

    return allmerged


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


def join_names_and_labels(idlabels, names):
    datadict = dict()
    for it in range(len(idlabels)):
        datadict[idlabels[it]] = names[it]
    return datadict


# =============================================================================
# Start of code
# =============================================================================
# Main code here
if __name__ == "__main__":

    plt.ion()

    # get the data
    printlog("Loading data...")
    rawdata = Table.read(WRITEPATH)

    # normalise the data
    printlog("Normalising data...")
    normdata = normalise_data(rawdata)

    # apply subset to data
    printlog("Constructing data matrix")
    if SUBSET:
        mask = get_random_choices(len(normdata), SUBSETSIZE)
    else:
        mask = np.ones(len(normdata['X']), dtype=bool)
    normdata = normdata[mask]

    # construct data matrix
    data = np.array([normdata['X'], normdata['Y'], normdata['Z'],
                     normdata['U'], normdata['V'], normdata['W']]).T
    # data = np.array([rawdata['X'], rawdata['Y'], rawdata['Z']]).T

    # get the true labels and group names
    labels_true = np.array(normdata['row'])
    groups = np.array(normdata['group'])

    # convert data to 32 bit
    data = np.array(data, dtype=np.float32)

    # ----------------------------------------------------------------------
    # DBscan example from :
    #      scikit-learn.org/stable/modules/clustering.html#dbscan
    #      http://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan
    #          .html#sphx-glr-auto-examples-cluster-plot-dbscan-py
    printlog("Calculating clustering using 'DBSCAN'...")
    start = time.time()

    sargs = dict(eps=0.05, min_samples=25)
    db = DBSCAN(**sargs).fit(data)
    end = time.time()
    # get mask and labels
    labels = db.labels_
    # report timing
    printlog('\t Time taken = {0} s'.format(end - start))

    # ----------------------------------------------------------------------
    # stats
    # Number of clusters in labels, ignoring noise if present.
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_clusters_true = len(set(labels_true)) - (1 if -1 in labels else 0)

    printlog('\t Estimated number of clusters: {0}'.format(n_clusters))
    # print stats
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
    merged = compare_results(groups, labels_true, labels)

    # ----------------------------------------------------------------------
    # Plot result
    printlog('Plotting graphs...')

    if PLOT_3D_ALL:
        # dont plot all results
        # mask = get_random_choices(len(data), 100000)
        mask = np.ones(len(data), dtype=bool)

        limits = plot_dims3d(data[mask], labels[mask], n_clusters, kind='out')

        limits = plot_dims3d(data[mask], labels_true[mask], n_clusters_true,
                             kind='in', setlimits=limits)

        plot_data3d(data[mask], setlimits=limits)

    if PLOT_2D_ALL:
        # dont plot all results
        # mask = get_random_choices(len(data), 100000)
        mask = np.ones(len(data), dtype=bool)

        limits = plot_dims(data[mask], labels[mask], n_clusters, kind='out')

        limits = plot_dims(data[mask], labels_true[mask], n_clusters_true,
                           kind='in', setlimits=limits)

        plot_data(data[mask])

    # ----------------------------------------------------------------------
    # plot merged
    if PLOT_3D_MERGED:
        umerged = np.unique(merged)
        if len(umerged) > 0:
            mask = []
            for group in groups:
                if group in umerged:
                    mask.append(True)
                else:
                    mask.append(False)
            mask = np.array(mask)

            # get names
            names_true = join_names_and_labels(labels_true, groups)
            # plot
            limits = plot_dims3d(data[mask], labels_true[mask], n_clusters_true,
                                 kind='in', names=names_true)
        else:
            print('error no clusters')

# =============================================================================
# End of code
# =============================================================================
