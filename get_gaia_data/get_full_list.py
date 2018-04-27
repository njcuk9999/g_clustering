#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# CODE NAME HERE

# CODE DESCRIPTION HERE

Created on 2018-04-26 at 16:42

@author: cook
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
from astropy import units as u
from tqdm import tqdm
import warnings


# =============================================================================
# Define variables
# =============================================================================

WORKSPACE = '/scratch/Projects/Gaia_clustering'
CHUNK_LOC = WORKSPACE + '/data/Gaia_5plx_chunks/'
SAVE_LOC = WORKSPACE + '/data/Gaia_5plx_chunks_processed/'
SAVE_COLS = ['designation', 'source_id', 'random_index',
             'ra', 'ra_error',
             'dec', 'dec_error',
             'parallax', 'parallax_error',
             'pmra', 'pmra_error',
             'pmdec', 'pmdec_error',
             'phot_g_mean_flux', 'phot_g_mean_flux_error',
             'phot_g_mean_mag',
             'phot_bp_mean_flux', 'phot_bp_mean_flux_error',
             'phot_bp_mean_mag',
             'phot_rp_mean_flux', 'phot_rp_mean_flux_error',
             'phot_rp_mean_mag',
             'radial_velocity', 'radial_velocity_error']
# -----------------------------------------------------------------------------


# =============================================================================
# Start of code
# =============================================================================
# Main code here
if __name__ == "__main__":
    # ----------------------------------------------------------------------
    # get chunk filenames
    print('\n Getting file list...')
    files = os.listdir(CHUNK_LOC)
    # create storage
    storage = dict()
    # loop around files
    print('\n Appending files to storage...')
    for f in tqdm(files):
        # construct path
        path = os.path.join(CHUNK_LOC, f)
        # read table
        with warnings.catch_warnings(record=True) as w:
            data = Table.read(path, format='votable')
        # loop around columns to keep
        for col in SAVE_COLS:
            if col not in storage:
                storage[col] = list(data[col])
            else:
                storage[col] += list(data[col])
        # delete data
        del data
    # finally convert dict to a table
    print('\n Converting storage to astropy.table')
    table = Table()
    for col in list(storage.keys()):
        table[col] = storage[col]
    # construct file name
    fullpath = os.path.join(SAVE_LOC, 'source_list_smallparams.fits')
    # save to file
    print('\n Saving table to file...')
    table.write(fullpath, overwrite=True)


# =============================================================================
# End of code
# =============================================================================
