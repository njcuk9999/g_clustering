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
import random
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import time
from astropy.table import Table
from scipy.stats import norm, chi2
from matplotlib.patches import Ellipse
import random
from tqdm import tqdm
import scipy.interpolate as interpolate
import mpl_scatter_density
from astrokin import convert

# =============================================================================
# Define variables
# =============================================================================
# Define paths
WORKSPACE = '/scratch/Projects/Gaia_clustering'
WRITEPATH = WORKSPACE + '/data/gaiadr2_plx5_rv_goodcut_eplx0.1.fits'
MVGROUPS = WORKSPACE + '/data/Gagne/Clusters.csv'
# -----------------------------------------------------------------------------
MARKERS = ['o', 'x', '+', '*']

DIMNAMES = ['X [pc]', 'Y [pc]', 'Z [pc]',
            'U [mas/yr]', 'V [mas/yr]', 'W [mas/yr]']
COLNAMES = ['ra', 'dec', 'distance', 'pmra', 'pmdec', 'radial_velocity']

PLOT_BACKGROUND = True
PLOT_2D_ALL = True


# =============================================================================
# Define functions
# =============================================================================
def get_region_data(filename):
    # load sim data
    simdata = Table.read(filename)
    # Getting columns from data
    name = np.array(simdata['Asso.'])
    X, Y = np.array(simdata['X']), np.array(simdata['Y'])
    Z = np.array(simdata['Z'])
    U, V = np.array(simdata['U']), np.array(simdata['V'])
    W = np.array(simdata['W'])
    sig00, sig11 = np.array(simdata['sig00']), np.array(simdata['sig11'])
    sig22, sig33 = np.array(simdata['sig22']), np.array(simdata['sig33'])
    sig44, sig55 = np.array(simdata['sig33']), np.array(simdata['sig55'])
    # construct matrix
    rdata = np.array([name, X, sig00, Y, sig11, Z, sig22, U, sig33, V, sig44,
                      W, sig55])
    # return
    return rdata


def normalise_data(data, mode='max'):
    newdata = np.zeros_like(data)
    if mode == 'max':
        for col in range(data.shape[1]):
            maxdata = np.max(data[:, col])
            newdata[:, col] = np.array(data[:, col])/maxdata
    else:
        for col in range(data.shape[1]):
            mean = np.mean(data[:, col])
            std = np.std(data[:, col])
            newdata[:, col] = (data[:, col] - mean)/std
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
        # if limits is not None:
        #     frame.set(xlim=limits[it][:2], ylim=limits[it][2:])
        # labels
        frame.set(xlabel='{0}'.format(DIMNAMES[r1]),
                  ylabel='{0}'.format(DIMNAMES[r2]))

    # title
    plt.suptitle('Data before clustering')

    # deal with blank frames
    for it in range(len(range1), nrows * ncols):
        frame = frames[pos[it][0]][pos[it][1]]
        frame.axis('off')


def plot_dims(d, idlabels, nclusters, kind='out', setlimits=None,
              regiondata=None):
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

        # plot known regions
        if regiondata is not None:
            frame = plot_known_regions(frame, regiondata, r1, r2)

        # set labels
        frame.set(xlabel='{0}'.format(DIMNAMES[r1]),
                  ylabel='{0}'.format(DIMNAMES[r2]))
        # set limits
        # if setlimits is None:
        #     frame.set(xlim=stats[:2], ylim=stats[2:])
        #     grphlimits.append(stats)
        # else:
        #     frame.set(xlim=setlimits[it][:2], ylim=setlimits[it][2:])
        #     grphlimits.append(setlimits[it])

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




def plot_known_regions(frame, regiondata, col1, col2):

    # loop around known regions
    for rdata in regiondata.T:
        name = rdata[0]
        # get centers and sigs
        cent1 = float(rdata[2 * col1 + 1])
        sig1 = float(rdata[2 * col1 + 2])
        cent2 = float(rdata[2 * col2 + 1])
        sig2 = float(rdata[2 * col2 + 2])
        # add ellipse
        frame = plot_ellipse(frame, cent1, cent2, sig1, sig2, colour='r')
        # frame.text(cent1, cent2, name)

    # return frame
    return frame


# =============================================================================
# Define Plot functions
# =============================================================================
# Taken from: https://stackoverflow.com/a/39749274/7858439
def cov_ellipse(cov, q=None, nsig=None, **kwargs):
    """
    Parameters
    ----------
    cov : (2, 2) array
        Covariance matrix.
    q : float, optional
        Confidence level, should be in (0, 1)
    nsig : int, optional
        Confidence level in unit of standard deviations.
        E.g. 1 stands for 68.3% and 2 stands for 95.4%.

    Returns
    -------
    width, height, rotation :
         The lengths of two axises and the rotation angle in degree
    for the ellipse.
    """
    if q is not None:
        q = np.asarray(q)
    elif nsig is not None:
        q = 2 * norm.cdf(nsig) - 1
    else:
        raise ValueError('One of `q` and `nsig` should be specified.')
    r2 = chi2.ppf(q, 2)

    val, vec = np.linalg.eigh(cov)
    width, height = 2 * np.sqrt(val[:, None] * r2)
    rotation = np.degrees(np.arctan2(*vec[::-1, 0]))

    return width, height, rotation


def printlog(message):
    message = message.split('\n')
    for mess in message:
        unix_time = time.time()
        human_time = time.strftime('%H:%M:%S', time.localtime(unix_time))
        dsec = int((unix_time - int(unix_time)) * 100)
        pargs = [BColors.OKGREEN, human_time, dsec, mess, BColors.ENDC]
        print('{0}{1}.{2:02d} | {3}{4}'.format(*pargs))


def plot_ellipse(frame, mu1, mu2, sig00, sig11, colour='k'):
    cov = np.diag([sig00, sig11])
    w, h, theta = cov_ellipse(cov, nsig=2.0)
    ell = Ellipse(xy=(mu1, mu2), width=w[0], height=h[0],
                  angle=theta, color=colour, zorder=5)
    ell.set_facecolor('none')
    frame.add_artist(ell)

    return frame


# defines the colours
class BColors:
    HEADER = '\033[95;1m'
    OKBLUE = '\033[94;1m'
    OKGREEN = '\033[92;1m'
    WARNING = '\033[93;1m'
    FAIL = '\033[91;1m'
    ENDC = '\033[0;0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


# =============================================================================
# Start of code
# =============================================================================
# Main code here
if __name__ == "__main__":

    plt.ion()

    # get the data
    printlog("Loading data...")
    rawdata = Table.read(WRITEPATH)

    # make distance column
    rawdata['distance'] = np.array(1000.0/rawdata['parallax'])
    # distance cut
    dmask = rawdata['distance'] < 175.0
    rawdata = rawdata[dmask]
    printlog('Distance cut, keep = {0}/{1}'.format(np.sum(dmask), len(dmask)))

    # colour cut
    g = np.array(rawdata['phot_g_mean_mag'])
    G = g - 5 * (np.log10(rawdata['distance']) - 1)
    bp = np.array(rawdata['phot_bp_mean_mag'])
    rp = np.array(rawdata['phot_rp_mean_mag'])
    #mask = G > (3.5 * (bp - rp) - 1.5)
    #mask &= G < (3.5 * (bp - rp) + 2.75)
    cmask = np.ones_like(g).astype(bool)
    rawdata = rawdata[cmask]
    printlog('Colour cut, keep = {0}/{1}'.format(np.sum(cmask), len(cmask)))

    # get the region data
    printlog("Reading region data...")
    region_data = get_region_data(MVGROUPS)

    # ----------------------------------------------------------------------
    # apply subset to data
    printlog("Constructing data matrix")

    # construct data matrix
    data = []
    for colname in COLNAMES:
        data.append(rawdata[colname])
    data = np.array(data).T


    # convert to X, Y, Z, U, V, W
    printlog("Converting data to XYZUVW")
    cdata = convert(ra=data[:, 0], dec=data[:, 1], distance=data[:, 2],
                    pmra=data[:, 3], pmde=data[:, 4], rv=data[:, 5])
    data = np.array(cdata).T

    # normalise the data
    printlog("Normalising data...")
    ndata = normalise_data(data)

    # convert data to 32 bit
    ndata = np.array(ndata, dtype=np.float64)

    # ----------------------------------------------------------------------
    # DBscan example from :
    #      scikit-learn.org/stable/modules/clustering.html#dbscan
    #      http://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan
    #          .html#sphx-glr-auto-examples-cluster-plot-dbscan-py
    printlog("Calculating clustering using 'DBSCAN'...")
    start = time.time()

    sargs = dict(eps=0.05, min_samples=50, algorithm='kd_tree')
    db = DBSCAN(**sargs).fit(ndata)
    end = time.time()

    # get mask and labels
    labels = db.labels_

    # ----------------------------------------------------------------------
    # stats
    # Number of clusters in labels, ignoring noise if present.
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    printlog('Number of clusters found = {0}'.format(n_clusters))

    # report timing
    printlog('\t Time taken = {0} s'.format(end - start))



    # ---------------------
    # ps-------------------------------------------------
    # Plot result
    printlog('Plotting graphs...')

    if PLOT_2D_ALL:
        # dont plot all results
        # mask = get_random_choices(len(data), 100000)
        mask = np.ones(len(data), dtype=bool)

        limits = plot_dims(data[mask], labels[mask], n_clusters, kind='out',
                           regiondata=region_data)

        plot_data(data[mask])

# =============================================================================
# End of code
# =============================================================================
