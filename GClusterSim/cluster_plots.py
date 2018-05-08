#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# CODE NAME HERE

# CODE DESCRIPTION HERE

Created on 2018-05-08 at 11:16

@author: cook
"""

import numpy as np
from scipy.stats import norm, chi2
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

import generic_functions


# =============================================================================
# Define variables
# =============================================================================
PRINT = generic_functions.printlog
EXIT = generic_functions.exit
MARKERS = ['o', 'x', '+', '*']
FIELD_LABELS = ['FIELD', '-1']
# -----------------------------------------------------------------------------

# =============================================================================
# Define user functions
# =============================================================================
def plot_graph(datamatrix, grouplabels, model, dimnames, dimlabels,
               plottitle=None):
    # set up the figure grid
    out = get_frames(model, dimnames, dimlabels)
    fig, frames, pos, col_labels, axis_labels = out
    # plot the data (with labels)
    frames = plot_data_labels(frames, datamatrix, grouplabels, pos,
                              col_labels, axis_labels)
    # plot the models
    frames = plot_models(frames, model, pos)
    # plot title
    if plottitle is not None:
        plt.suptitle(plottitle)
    # set up the figure grid
    out = get_frames(model, dimnames, dimlabels)
    fig, frames, pos, col_labels, axis_labels = out
    # plot the data
    frames = plot_data(frames, datamatrix, pos, col_labels, axis_labels)
    # plot title
    if plottitle is not None:
        plt.suptitle(plottitle)
    # show and close
    plt.show()
    plt.close()


def get_frames(models, dimnames, dimlabels):

    # get shape of models
    shape = len(models['CENT'][0])
    # check that the number of labels matches the shape
    if len(dimnames) != shape:
        wmsg = 'Error: model shape ({0}) does not match dimnames shape ({1})'
        PRINT(wmsg.format(shape, len(dimnames)), level='error')
        EXIT(1)
    # get optimal grid
    nrows, ncols, positions = optimal_grid(shape)
    # get ranges for graph plotting
    range1 = list(range(shape - 1))
    range2 = list(range(1, shape))
    # construct axis labels
    axislabels, colnames = [], []
    for it in range(len(range1)):
        # get positions
        r1, r2 = range1[it], range2[it]
        # get axis labels
        colnames.append([dimnames[r1], dimnames[r2]])
        axislabels.append([dimlabels[r1], dimlabels[r2]])
    # set up figure
    fig, frames = plt.subplots(nrows=nrows, ncols=ncols)
    # return fig and frames
    return fig, frames, positions, colnames, axislabels



def plot_data_labels(frames, data, labels, positions, colnames, axislabels):

    # get the unique labels
    ulabels = np.unique(labels)
    # get number of axes
    naxes = len(axislabels)
    # get ranges for graph plotting
    range1 = list(range(naxes - 1))
    range2 = list(range(1, naxes))
    # get colour marker combinations
    colors = [plt.cm.jet(each)
              for each in np.linspace(0, 1, len(ulabels))]
    marks = np.random.choice(MARKERS, size=len(ulabels))
    # loop around each axes
    for it in range(naxes):
        # get positions of dimensions in data
        frame = frames[positions[it][0]][positions[it][1]]
        # loop around labels and plot
        for uit, ulabel in enumerate(ulabels):
            # label mask
            lmask = labels == ulabel
            # get x-axis and y-axis data
            xdata = data[colnames[it][0]][lmask]
            ydata = data[colnames[it][1]][lmask]

            # plot if field
            if str(ulabel).upper() in FIELD_LABELS:
                frame.plot(xdata, ydata,
                           marker='.', color='k',
                           linestyle='none', markersize=1)
            # plot
            else:
                frame.plot(xdata, ydata,
                           marker=marks[uit], color=tuple(colors[uit]),
                           linestyle='none', markersize=2)
        # set axis labels
        frame.set(xlabel=axislabels[it][0],
                  ylabel=axislabels[it][1])

    # deal with blank frames
    for it in range(naxes, frames.size):
        frame = frames[positions[it][0]][positions[it][1]]
        frame.axis('off')
    # return frames
    return frames


def plot_data(frames, data, positions, colnames, axislabels):
    # get number of axes
    naxes = len(axislabels)
    # loop around each axes
    for it in range(naxes):
        # get positions of dimensions in data
        frame = frames[positions[it][0]][positions[it][1]]
        # get x-axis and y-axis data
        xdata = data[colnames[it][0]]
        ydata = data[colnames[it][1]]
        # plot data
        frame.plot(xdata, ydata,
                   marker='.', color='k',
                   linestyle='none', markersize=2)
        # set axis labels
        frame.set(xlabel=axislabels[it][0],
                  ylabel=axislabels[it][1])
    # deal with blank frames
    for it in range(naxes, frames.size):
        frame = frames[positions[it][0]][positions[it][1]]
        frame.axis('off')
    # return frames
    return frames


def plot_models(frames, models, positions):
    # get number of axes
    naxes = len(positions)
    # get ranges for graph plotting
    range1 = list(range(naxes - 1))
    range2 = list(range(1, naxes))
    # construct axis labels
    for it in range(len(range1)):
        # get positions
        r1, r2 = range1[it], range2[it]
        # get positions of dimensions in data
        frame = frames[positions[it][0]][positions[it][1]]
        # loop around models
        for mit, model in enumerate(models):

            if model['NAME'] == 'FIELD':
                continue
            # plot ellipse
            kwargs = dict(mus=model['CENT'], cov=model['COV'],
                          axis0=r1, axis1=r2, colour='r')
            frame = plot_2d_ellipse_from_full(frame, **kwargs)

    return frames


# =============================================================================
# Define internal functions
# =============================================================================
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


def plot_2d_ellipse_from_full(frame, mus, cov, axis0=0, axis1=1, **kwargs):

    # get means
    mu1 = mus[axis0]
    mu2 = mus[axis1]
    # get 2D covariance matrix
    covsmall = [[cov[axis0, axis0], cov[axis0, axis1]],
                [cov[axis1, axis0], cov[axis1, axis1]]]
    # plot 2D ellipse
    return plot_2d_ellipse(frame, mu1, mu2, covsmall, **kwargs)


def plot_2d_ellipse(frame, mu1, mu2, cov, colour='k'):
    w, h, theta = cov_ellipse(cov, nsig=2.0)
    ell = Ellipse(xy=(mu1, mu2), width=w[0], height=h[0],
                  angle=theta, color=colour, zorder=5)
    ell.set_facecolor('none')
    frame.add_artist(ell)
    return frame


# =============================================================================
# Define Math functions
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
