#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# CODE NAME HERE

# CODE DESCRIPTION HERE

Created on 2018-03-02 at 12:33

@author: cook
"""
from astropy.table import Table
from astrokin.astrokin import convert

# =============================================================================
# Define variables
# =============================================================================
# Define paths
WORKSPACE = '/scratch/Projects/Gaia_clustering'
SIMPATH = WORKSPACE + '/data/Sim/Simulation_simple.fits'
# -----------------------------------------------------------------------------

# =============================================================================
# Define functions
# =============================================================================


# =============================================================================
# Start of code
# =============================================================================
# Main code here
if __name__ == "__main__":
    # ----------------------------------------------------------------------
    # load data
    print('Loading data {0}...'.format(SIMPATH))
    data = Table.read(SIMPATH)

    print('Running conversion...')
    output = convert(x=data['X'], y=data['Y'], z=data['Z'],
                     vu=data['U'], vv=data['V'], vw=data['W'])

    print('Adding back to table...')
    data['ra'] = output[0]
    data['dec'] = output[1]
    data['dist'] = output[2]
    data['pmra'] = output[3]
    data['pmde'] = output[4]
    data['rv'] = output[5]

    print('Saving back to file...')
    data.write(SIMPATH, overwrite=True)


# =============================================================================
# End of code
# =============================================================================
