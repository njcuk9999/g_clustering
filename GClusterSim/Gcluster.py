#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gcluster.py mode=INT filename=STR

    Options are:
        mode = 0: use DBSCAN (default)
        mode = 1: use HDBSCAN

        filenum = 0 'sim.fits' (default, simulated data)
        filenum = 1 'GDR2_eplx0.1.fits' (Gaia DR2 data)

        filled by "FILES" list (below)

Created on 2018-01-16 at 14:13

@author: cook
"""
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy import units as un
import time
import os
import sys

import cluster_plots
import generic_functions
from astrokin import convert

# =============================================================================
# Define variables
# =============================================================================
# constnat functions
PRINT = generic_functions.printlog
EXIT = generic_functions.exit
# Define paths
WORKSPACE = '/scratch/Projects/Gaia_clustering/'
PARAM_FILE = WORKSPACE + 'data/Gagne/banyan_sigma_parameters.fits'
DATA_PATH = WORKSPACE + 'data/ClusterRunData/'
PLOT_PATH = WORKSPACE + 'plots/Gcluster/simulation.fits'
FILES = ['simulationNS300000_NC3000.fits',
         'GDR2_eplx0.1.fits']
# -----------------------------------------------------------------------------
DIMNAMES = ['X', 'Y', 'Z', 'U', 'V', 'W']
AXISNAMES = ['X [pc]', 'Y [pc]', 'Z [pc]',
             'U [mas/yr]', 'V [mas/yr]', 'W [mas/yr]']
COLNAMES1 = ['ra', 'dec', 'distance', 'pmra', 'pmdec', 'radial_velocity']
COLNAMES2 = ['ra', 'dec', 'parallax', 'pmra', 'pmdec', 'radial_velocity']

COLS = dict(ra='ra', dec='dec', distance='distance', pmra='pmra',
            pmde='pmdec', rv='radial_velocity')
PLXCOL = ['PLX', 'PARALLAX']
PLXUNITS = un.mas
# -----------------------------------------------------------------------------
# Masks
DIST_MASK = True
DISTANCE_CUT = 200.0
COLOUR_MASK = False
# -----------------------------------------------------------------------------
# clustering
EPS = 0.05
MINSAMPLES = 10

# =============================================================================
# Define functions
# =============================================================================
def get_arguments():
    kwargs = dict()
    for arg in sys.argv:
        if '=' in arg:
            try:
                key, value = arg.split('=')
                kwargs[key] = value
            except ValueError:
                PRINT('Argument: {0} not understood'.format(arg), level='error')
                EXIT(1)
    # assign mode from kwargs
    try:
        mode = int(kwargs.get('mode', 0))
    except ValueError:
        PRINT('mode = {0} not understood'.format(kwargs['mode']), level='error')
        EXIT(1)
        mode = 0
    # assign filename from kwargs
    try:
        filenumber = int(kwargs.get('filenum', 0))
        filepath = os.path.join(DATA_PATH, FILES[filenumber])
    except ValueError:
        PRINT('filenum = {0} not understood'.format(kwargs['filenum']),
              level='error')
        EXIT(1)
        filepath = os.path.join(DATA_PATH, FILES[0])
    if not os.path.exists(filepath):
        PRINT('Error: file={0} not found'.format(filepath))
        EXIT(1)
    # return filepath
    return mode, filepath


def get_parameters():
    # get file name
    filename = PARAM_FILE
    # read parameters
    parameters_str = Table.read(filename, format='fits')
    # Remove white spaces in names
    names = np.array(parameters_str['NAME'])
    parameters_str['NAME'] = np.chararray.strip(names)
    # Create new table with only the data required
    keep = Table()
    keep['NAME'] = parameters_str['NAME']
    keep['CENT'] = parameters_str['CENTER_VEC']
    keep['COV'] = parameters_str['COVARIANCE_MATRIX']
    # return keep
    return keep


def process_data(raw_data):

    # validate data
    # -------------------------------------------------------------------------
    data_type = 0
    # Data type 1: we have all "DIMNAMES"
    if data_type == 0:
        cond = True
        for dimname in DIMNAMES:
            if dimname not in raw_data.colnames:
                cond &= False
        if cond:
            tabledata = raw_data
            tabledata = calc_distance(tabledata)
            data_type = 1
            colnames = DIMNAMES
    # -------------------------------------------------------------------------
    if data_type == 0:
        # Data type 2: we have all "COLNAMES1"
        cond = True
        for colname in COLNAMES2:
            if colname not in raw_data.colnames:
                cond &= False
        if cond:
            tabledata = convert_from_ra_dec(raw_data)
            data_type = 2
            colnames = COLNAMES2
    # -------------------------------------------------------------------------
    if data_type == 0:
        # Data type 3: we have all "COLNAMES2"
        cond = True
        for colname in COLNAMES2:
            if colname not in raw_data.colnames:
                cond &= False
        if cond:
            tabledata = raw_data
            tabledata = calc_distance(tabledata)
            tabledata = convert_from_ra_dec(tabledata)
            data_type = 3
            colnames = COLNAMES2

    PRINT('Data type "{0}" processed'.format(data_type))

    return tabledata, colnames


def apply_masks(tabledata):
    # -------------------------------------------------------------------------
    # DISTANCE MASK
    # -------------------------------------------------------------------------
    if DIST_MASK:
        # distance cut
        dmask = np.array(tabledata[COLS['distance']]) < DISTANCE_CUT
        tabledata = tabledata[dmask]
        PRINT('Distance cut, keep = {0}/{1}'.format(np.sum(dmask), len(dmask)))

    # return masked data
    return tabledata


def convert_from_ra_dec(tabledata):
    PRINT("Converting data to XYZUVW")
    kwargs = dict()
    for col in COLS:
        kwargs[col] = tabledata[COLS[col]]
    cdata = convert(**kwargs)
    # add cdata to new table
    outtable = Table()
    for cc_it, col in enumerate(DIMNAMES):
        outtable[col] = cdata[:, cc_it]
    return outtable


def calc_distance(tabledata):
    dist_col = COLS['distance']
    # -------------------------------------------------------------------------
    # check for parallax column
    has_parallax = False
    parallax_col = None
    for pcol in PLXCOL:
        if pcol in tabledata.colnames:
            has_parallax = True
            parallax_col = pcol
    # calculate distance from parallax
    if has_parallax:
        parallax = np.array(tabledata[parallax_col]) * PLXUNITS
        tabledata[dist_col] = 1.0/parallax.to(un.arcsec).value
    # -------------------------------------------------------------------------
    # check for X, Y, Z column
    cond1 = 'X' in tabledata.colnames
    cond2 = 'Y' in tabledata.colnames
    cond3 = 'Z' in tabledata.colnames
    if cond1 & cond2 & cond3:
        x = tabledata['X']
        y = tabledata['Y']
        z = tabledata['Z']
        tabledata[dist_col] = np.sqrt(x**2 + y**2 + z**2)
    # -------------------------------------------------------------------------
    return tabledata


def normalise_data(data_matrix, mode='max'):
    newdata = np.zeros_like(data_matrix)
    if mode == 'max':
        for col in range(data_matrix.shape[1]):
            maxdata = np.max(data_matrix[:, col])
            newdata[:, col] = np.array(data_matrix[:, col])/maxdata
    else:
        for col in range(data_matrix.shape[1]):
            mean = np.mean(data_matrix[:, col])
            std = np.std(data_matrix[:, col])
            newdata[:, col] = (data_matrix[:, col] - mean)/std
    return newdata


def run_clustering(data_matrix, mode=0):
    # ----------------------------------------------------------------------
    if mode == 0:
        from sklearn.cluster import DBSCAN
        # DBscan example from :
        #      scikit-learn.org/stable/modules/clustering.html#dbscan
        #      http://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan
        #          .html#sphx-glr-auto-examples-cluster-plot-dbscan-py
        PRINT("Calculating clustering using 'DBSCAN'...")
        PRINT('\tEPS={0} MINSAMPLES={1}'.format(EPS, MINSAMPLES))
        start = time.time()
        sargs = dict(eps=EPS, min_samples=MINSAMPLES, algorithm='kd_tree')
        db = DBSCAN(**sargs).fit(data_matrix)
        # get mask and labels
        clabels = db.labels_
        end = time.time()
    # ----------------------------------------------------------------------
    elif mode == 1:
        import hdbscan
        PRINT("Calculating clustering using 'HDBSCAN'...")
        PRINT('\tMINSAMPLES={0}'.format(MINSAMPLES))
        start = time.time()
        clusterer = hdbscan.HDBSCAN(min_cluster_size=MINSAMPLES)
        clabels = clusterer.fit_predict(data_matrix)
        end = time.time()
    # ----------------------------------------------------------------------
    else:
        PRINT('Error: mode not supported.', level='error')
        EXIT(1)
        clabels, start, end = None, 0, 0
    # ----------------------------------------------------------------------
    # report timing
    PRINT('\t Time taken = {0} s'.format(end - start))
    # return labels
    return clabels


# =============================================================================
# Start of code
# =============================================================================
# Main code here
if __name__ == "__main__":
    # ----------------------------------------------------------------------
    # get mode
    pmode, pfilepath = get_arguments()
    # ----------------------------------------------------------------------
    # get the data
    PRINT("Loading data...")
    rawdata = Table.read(pfilepath)
    # ----------------------------------------------------------------------
    # process the data
    PRINT("Processing data...")
    tdata, column_names = process_data(rawdata)
    # ----------------------------------------------------------------------
    # apply any masks
    tdata = apply_masks(tdata)
    # ----------------------------------------------------------------------
    # construct data matrix
    PRINT("Constructing data matrix")
    data = []
    for col_name in column_names:
        data.append(tdata[col_name])
    data = np.array(data).T
    # ----------------------------------------------------------------------
    # normalise the data
    PRINT("Normalising data...")
    ndata = normalise_data(data)
    # ----------------------------------------------------------------------
    # convert data to 32 bit
    ndata = np.array(ndata, dtype=np.float64)
    # ----------------------------------------------------------------------
    # Run clustering
    labels = run_clustering(ndata, mode=pmode)
    # ----------------------------------------------------------------------
    # stats
    # Number of clusters in labels, ignoring noise if present.
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    title = 'Number of clusters found = {0}'.format(n_clusters)
    PRINT(title)
    # ----------------------------------------------------------------------
    # get parameter table
    PRINT('Getting parameters from file...')
    params = get_parameters()
    # ----------------------------------------------------------------------
    # Plot result
    PRINT('Plotting graphs...')
    table = Table()
    table['group'] = labels
    for c_it, column in enumerate(DIMNAMES):
        table[column] = data[:, c_it]
    # ----------------------------------------------------------------------
    # plot graph
    cluster_plots.plot_graph(table, table['group'], params, DIMNAMES,
                             AXISNAMES, plottitle=title)
    # ----------------------------------------------------------------------
    PRINT('Clustering completed successfully.')

# =============================================================================
# End of code
# =============================================================================
