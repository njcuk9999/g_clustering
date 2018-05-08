#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# CODE NAME HERE

# CODE DESCRIPTION HERE

Created on 2018-05-08 at 10:52

@author: cook
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy import units as u
from tqdm import tqdm

import cluster_plots
import generic_functions

# =============================================================================
# Define variables
# =============================================================================
WORKSPACE = '/scratch/Projects/Gaia_clustering/'
PARAM_FILE = WORKSPACE + 'data/Gagne/banyan_sigma_parameters.fits'
WRITE_FILE = WORKSPACE + 'data/Gagne/simulation.fits'
PLOT_PATH = WORKSPACE + 'plots/Gcluster/simulation.fits'
PRINT = generic_functions.printlog
COLNAMES = ['X', 'Y', 'Z', 'U', 'V', 'W']
AXISNAMES = ['X [pc]', 'Y [pc]', 'Z [pc]',
             'U [mas/yr]', 'V [mas/yr]', 'W [mas/yr]']
# cluster properties
DISTANCE_LIMIT = 200.0
NSTARS = 300000
NCLUSTERSTARS = 3000
MINNUMBER = 25
MAXNUMBER = 1000
# -----------------------------------------------------------------------------


# =============================================================================
# Define functions
# =============================================================================
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


def get_numbers(keep):

    # get number of clusters
    nclusters = len(np.unique(keep['NAME'])) - 1

    # check that MAXNUMBER * nclusters is greater than NCLUSTER
    if MAXNUMBER * nclusters < NCLUSTERSTARS:
        emsg = ('Error: "MAXNUMBER"={0} too low to generate "NCLUSTER"={1}'
                ' clusters (Number of clusters = {2}')
        PRINT(emsg.format(MAXNUMBER, NCLUSTERSTARS, nclusters))
    # generate cluster numbers
    cond, cnumbers = True, []
    while cond:
        # get raw numbers
        raw = np.random.uniform(MINNUMBER, MAXNUMBER*2, size=nclusters)
        # get the actual numbers
        cnumbers = NCLUSTERSTARS * raw//np.sum(raw)
        # check numbers are correct
        cond1 = np.min(cnumbers) > MINNUMBER
        cond2 = np.max(cnumbers) < MAXNUMBER
        # top up the number to NCLUSTER
        while np.sum(cnumbers) != NCLUSTERSTARS:
            index = np.random.choice(range(len(cnumbers)))
            cnumbers[index] += 1
        # if conditions met break loop
        if cond1 & cond2:
            cond = False
    # assign these to a cluster
    pop_numbers = dict()
    for n_it, pop_name in enumerate(params['NAME']):
        if str(pop_name).upper() == 'FIELD':
            continue
        else:
            pop_numbers[pop_name] = int(cnumbers[n_it])
    # add field star numbers
    pop_numbers['FIELD'] = NSTARS - NCLUSTERSTARS
    # return the numbers
    return pop_numbers


def add_field(data, number=10000):
    print('Field number = {0}'.format(number))
    return data


def save_sim(datadict, n_clust, n_clusterstar, n_field):
    extension = 'NC_{0}_NCS{1}_NF{2}'.format(n_clust, n_clusterstar, n_field)
    filename = WRITE_FILE.replace('.fits', extension + '.fits')
    simtable = Table()
    for column in datadict:
        simtable[column] = datadict[column]
    simtable.write(filename, format='fits', overwrite=True)


# =============================================================================
# Define Math functions
# =============================================================================
def multivariate_gaussian(mu, cov, number=10000):
    return np.random.multivariate_normal(mu, cov, size=number)


# =============================================================================
# Start of code
# =============================================================================
# Main code here
if __name__ == "__main__":
    # ----------------------------------------------------------------------
    # get parameter table
    PRINT('Getting parameters from file...')
    params = get_parameters()
    # decide on number of objects
    numbers = get_numbers(params)
    # storage of variables
    sims = dict(X=[], Y=[], Z=[], U=[], V=[], W=[], group=[], row=[])
    # loop around groups (in params['NAME']) and set sim data
    PRINT('Generating fake data from parameters...')
    nclust, nclusterstar, nfield = 0, 0, 0
    for row in range(len(params)):
        # get the centers and covariance for each cluster
        name = params['NAME'][row]
        centers, covarr = params['CENT'][row], params['COV'][row]
        # progress
        PRINT('\t Creating {0}'.format(name))
        # deal with field
        if name == 'FIELD':
            # nfield += numbers[name]
            continue
        else:
            num = numbers[name]
            data_row = multivariate_gaussian(centers, covarr, number=num)
            nclust += 1
            nclusterstar += num
        # add field data
        data_row = add_field(data_row, number=numbers['FIELD'])
        # store
        sims['row'] = np.append(sims['row'], np.repeat([row], num))
        for c_it, col in enumerate(COLNAMES):
            sims[col] = np.append(sims[col], data_row[:, c_it])
        sims['group'] = np.append(sims['group'], np.repeat([name], num))
    # save to file
    PRINT('Saving to file...')
    save_sim(sims, nclust, nclusterstar, nfield)
    # plot graph
    PRINT('Plotting graphs...')
    title = ('Simulation Nclusters = {0} NClusterStars = {1}, Nstars = {2}'
             ''.format(nclust, nclusterstar, nfield))
    cluster_plots.plot_graph(sims, sims['group'], params, COLNAMES, AXISNAMES,
                             plottitle=title)


# =============================================================================
# End of code
# =============================================================================
