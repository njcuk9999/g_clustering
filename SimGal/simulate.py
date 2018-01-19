#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# CODE NAME HERE

# CODE DESCRIPTION HERE

Created on 2018-01-16 at 11:04

@author: cook



Version 0.0.0
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from scipy.stats import norm, chi2
from matplotlib.patches import Ellipse
import random
from tqdm import tqdm
import scipy.interpolate as interpolate
import mpl_scatter_density
from astropy.visualization import LogStretch
from astropy.visualization.mpl_normalize import ImageNormalize


# =============================================================================
# Define variables
# =============================================================================

# number of stars in simulation
N_STARS = 1000000
# fraction of stars in moving groups
N_MOVING = 10000
# maximum distance of field stars [pc]
FIELD_DISTANCE = 200

# Define paths
WORKSPACE = '/scratch/Projects/Gaia_clustering'
MVGROUPS = WORKSPACE + '/data/Gagne/Clusters.csv'
WRITEPATH = WORKSPACE + '/data/Sim/Simulation_simple.fits'


# -----------------------------------------------------------------------------

# =============================================================================
# Define group functions
# =============================================================================
def read_groups(filename):
    data = Table.read(filename)
    return data


def get_group_sizes(groupsize, n_stars):
    # get raw numbers between 0 and 1
    raw = np.random.uniform(0, 1, size=groupsize)
    # get sizes of each number (that sum to sum)
    new = n_stars * raw/np.sum(raw)
    # convert to integers
    groupsizes = np.array(np.round(new, 0), dtype=int)
    # return groupsizes and true total
    return groupsizes, np.sum(groupsizes)


def sample_from_truncated_gaussian(median, fwhm, num, low=np.inf, high=np.inf):

    g = np.random.normal(median, fwhm, size=num*10)
    # find those outside truncation
    mask = (g < low) | (g > high)
    # remove those outside truncation
    g = g[~mask]
    # use distribution to get random samples
    return inverse_transform_sampling(g, int(num/10), num)


def inverse_transform_sampling(data, n_bins=40, n_samples=1000):
    hist, bin_edges = np.histogram(data, bins=n_bins, density=True)
    cum_values = np.zeros(bin_edges.shape)
    cum_values[1:] = np.cumsum(hist * np.diff(bin_edges))
    inv_cdf = interpolate.interp1d(cum_values, bin_edges)
    r = np.random.rand(n_samples)
    return inv_cdf(r)


def generate_field_population(fieldsize):

    xfield = np.random.uniform(-FIELD_DISTANCE, FIELD_DISTANCE, size=fieldsize)
    yfield = np.random.uniform(-FIELD_DISTANCE, FIELD_DISTANCE, size=fieldsize)

    # TODO: should be from probability distributions
    rargs = [0.0, 1500/3.0, fieldsize, -FIELD_DISTANCE, FIELD_DISTANCE]
    zfield = sample_from_truncated_gaussian(*rargs)

    # TODO: should be from probability distributions
    ufield = np.random.normal(loc=0.0, scale=200/3.0, size=fieldsize)

    # TODO: should be from probability distributions
    vfield = np.random.normal(loc=-50.0, scale=100/3.0, size=fieldsize)

    # TODO: should be from probability distributions
    wfield = np.random.normal(loc=0.0, scale=100/3.0, size=fieldsize)

    return xfield, yfield, zfield, ufield, vfield, wfield


def save_population(data):

    table = Table()

    table['row'] = data['row']
    table['X'] = data['X']
    table['Y'] = data['Y']
    table['Z'] = data['Z']
    table['U'] = data['U']
    table['V'] = data['V']
    table['W'] = data['W']
    table['group'] = data['group']

    table.write(WRITEPATH, overwrite=True)


# =============================================================================
# Define Math functions
# =============================================================================
def multivariate_gaussian(means, sigs, number=10000):
    mu = np.array(means)
    if len(sigs.shape) == 1:
        cov = np.diag(sigs)
    else:
        cov = np.array(sigs)
    return np.random.multivariate_normal(mu, cov, size=number)


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


def plot_2D(data, xaxis='X', yaxis='Y', sigxaxis='sig00', sigyaxis='sig11',
            frame=None):
    # get columns
    name = np.array(data['Asso.'])
    axis1 = np.array(data[xaxis])
    axis2 = np.array(data[yaxis])
    eaxis1 = np.array(data[sigxaxis])
    eaxis2 = np.array(data[sigyaxis])
    # deal with no frame
    if frame is None:
        plot = True
        fig, frame = plt.subplots(ncols=1, nrows=1)
    else:
        plot = False
    # loop around groups
    for row in range(len(data)):
        frame = plot_ellipse(frame, axis1[row], axis2[row], eaxis1[row],
                             eaxis2[row], colour='r')
        # frame.plot(Y[row], X[row], color='k', marker='x', ms=3)
        frame.text(axis1[row], axis2[row], s=name[row],
                   horizontalalignment='center',
                   verticalalignment='center', color='g')
    # finalise
    if plot:
        plt.show()
        plt.close()
    else:
        return frame


def get_random_choices(array, num):
    mask = random.choices(range(len(array)), k=num)
    return mask


def plot_some_rows(sims, xaxis='X', yaxis='Y', nbins=100, fig=None, frame=None):



    norm = ImageNormalize(vmin=0., vmax=1000, stretch=LogStretch())


    # deal with no frame
    if frame is None:
        plot = True
        fig = plt.figure()
        frame = fig.add_subplot(1, 1, 1, projection='scatter_density')
    else:
        plot = False
    # get columns
    rows = np.array(sims['row'])
    x = np.array(sims[xaxis])
    y = np.array(sims[yaxis])

    density = frame.scatter_density(x, y, norm=norm)

    fig.colorbar(density, label='Number of points per pixel',
                 ax=frame)

    # finalise
    if plot:
        plt.show()
        plt.close()
    else:
        return frame


def plot_ellipse(frame, mu1, mu2, sig00, sig11, colour='k'):
    cov = np.diag([sig00, sig11])
    w, h, theta = cov_ellipse(cov, nsig=2.0)
    ell = Ellipse(xy=(mu1, mu2), width=w[0], height=h[0],
                  angle=theta, color=colour)
    ell.set_facecolor('none')
    frame.add_artist(ell)

    return frame


# =============================================================================
# Start of code
# =============================================================================
# Main code here
if __name__ == "__main__":
    # ----------------------------------------------------------------------
    # load sim data
    print('\n Loading sim table...')
    simdata = read_groups(MVGROUPS)

    # -------------------------------------------------------------------------
    # Getting columns from data
    name = np.array(simdata['Asso.'])
    X, Y = np.array(simdata['X']), np.array(simdata['Y'])
    Z = np.array(simdata['Z'])
    U, V = np.array(simdata['U']), np.array(simdata['V'])
    W = np.array(simdata['W'])
    sig00, sig11 = np.array(simdata['sig00']), np.array(simdata['sig11'])
    sig22, sig33 = np.array(simdata['sig22']), np.array(simdata['sig33'])
    sig44, sig55 = np.array(simdata['sig33']), np.array(simdata['sig55'])

    # -------------------------------------------------------------------------
    # work out the number of objects in each group

    f_group = N_MOVING/N_STARS
    total_in_groups = int(f_group * N_STARS)
    num_groups = len(simdata)

    # create the numbers in each group
    gsize, total_in_groups = get_group_sizes(num_groups, total_in_groups)
    total_in_field = N_STARS - total_in_groups

    # storage of variables
    sims = dict(X=[], Y=[], Z=[], U=[], V=[], W=[], group=[], row=[])

    # -------------------------------------------------------------------------
    # loop around each group and create them
    print('\n Creating moving groups...')
    for row in tqdm(range(len(simdata))):
        # get means and covariance matrix
        means_row = np.array([X[row], Y[row], Z[row], U[row], V[row], W[row]])
        sigs_row = np.array([sig00[row], sig11[row], sig22[row], sig33[row],
                             sig44[row], sig55[row]])
        # create multigaus distribution
        data_row = multivariate_gaussian(means_row, sigs_row, gsize[row])
        # store in storage
        sims['row'] += list(np.repeat([row], gsize[row]))
        sims['X'] += list(data_row[:, 0])
        sims['Y'] += list(data_row[:, 1])
        sims['Z'] += list(data_row[:, 2])
        sims['U'] += list(data_row[:, 3])
        sims['V'] += list(data_row[:, 4])
        sims['W'] += list(data_row[:, 5])
        sims['group'] += list(np.repeat([name[row]], gsize[row]))

    # -------------------------------------------------------------------------
    # create remaining stars (field)
    print('\n Creating field population...')
    data_field = generate_field_population(total_in_field)

    # add to sims
    sims['row'] += list(np.repeat([-1], total_in_field))
    sims['X'] += list(data_field[0])
    sims['Y'] += list(data_field[1])
    sims['Z'] += list(data_field[2])
    sims['U'] += list(data_field[3])
    sims['V'] += list(data_field[4])
    sims['W'] += list(data_field[5])
    sims['group'] += list(np.repeat('FIELD', total_in_field))


    # -------------------------------------------------------------------------
    print('\n Plotting graph...')

    groups = [['Y', 'X'], ['X', 'Z'], ['U', 'V'], ['U', 'W']]
    grouplabels = [['Y [pc]', 'X [pc]'], ['X [pc]', 'Z [pc]'],
                   ['U [mas/yr]', 'V [mas/yr]'],
                   ['U [mas/yr]', 'W [mas/yr]']]
    groupsigs = [['sig11', 'sig00'], ['sig00', 'sig22'],
                 ['sig33', 'sig44'], ['sig33', 'sig44']]
    grouppos = [[0, 0], [0, 1], [1, 0], [1, 1]]

    # set up figure
    fig = plt.figure()
    frames = [fig.add_subplot(2, 2, 1, projection='scatter_density'),
              fig.add_subplot(2, 2, 2, projection='scatter_density'),
              fig.add_subplot(2, 2, 3, projection='scatter_density'),
              fig.add_subplot(2, 2, 4, projection='scatter_density')]

    # loop around plot groups
    for g_it in tqdm(range(len(groups))):
        # get this iterations values
        gax1, gax2 = groups[g_it]
        egax1, egax2 = groupsigs[g_it]
        gax1label, gax2label = grouplabels[g_it]
        # get this iterations frame
        frame = frames[g_it]
        # plot sim data
        frame = plot_some_rows(sims, gax1, gax2, 100, fig, frame)
        # plot group boundaries and names
        frame = plot_2D(simdata, gax1, gax2, egax1, egax2, frame)
        # finalise and show
        frame.set(xlabel='{0}'.format(gax1label),
                  ylabel='{0}'.format(gax2label))
    # show and close
    plt.show()
    plt.close()

    # -------------------------------------------------------------------------
    print('\n Saving results to file...')
    save_population(sims)

# =============================================================================
# End of code
# =============================================================================
