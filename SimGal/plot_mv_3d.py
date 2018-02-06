#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# CODE NAME HERE

# CODE DESCRIPTION HERE

Created on 2018-02-06 at 15:30

@author: cook



Version 0.0.0
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
from astropy import units as u
from tqdm import tqdm
import warnings
from mpl_toolkits.mplot3d import Axes3D

import plot
import sys
if sys.version_info.major == 2:
    from mayavi import mlab
else:
    mlab = None



# =============================================================================
# Define variables
# =============================================================================
# Define paths
WORKSPACE = '/scratch/Projects/Gaia_clustering'
MVGROUPS = WORKSPACE + '/data/Gagne/Clusters.csv'
WRITEPATH = WORKSPACE + '/data/Sim/Simulation_simple.fits'
# -----------------------------------------------------------------------------

HIGHLIGHTED = ['ROPH', 'USCO', 'UCRA', 'CRA']
USE_SELECTION = True

# =============================================================================
# Define functions
# =============================================================================
def read_groups(filename):
    data = Table.read(filename)
    return data

# =============================================================================
# Start of code
# =============================================================================
# Main code here
if __name__ == "__main__":
    # ----------------------------------------------------------------------
    # ----------------------------------------------------------------------
    # load sim data
    print('\n Loading sim table...')
    sdata = read_groups(MVGROUPS)

    plt.ion()
    # -------------------------------------------------------------------------
    # Getting columns from data
    names = np.array(sdata['Asso.'])
    X, Y, Z = sdata['X'], sdata['Y'], sdata['Z']
    U, V, W = sdata['U'], sdata['V'], sdata['W']
    eX, eY, eZ = sdata['sig00'], sdata['sig11'], sdata['sig22']
    eU, eV, eW = sdata['sig33'], sdata['sig44'], sdata['sig55']


    # get colours
    colours = plot.gen_rgb_colors(len(names))

    # -------------------------------------------------------------------------
    # fig = plt.figure(figsize=plt.figaspect(1))  # Square figure
    # frame1 = fig.add_subplot(121, projection='3d')
    # frame2 = fig.add_subplot(122, projection='3d')

    fig1 = mlab.figure(bgcolor=(0.0, 0.0, 0.0))
    fig2 = mlab.figure(bgcolor=(0.0, 0.0, 0.0))

    for it in range(len(names)):

        if names[it] not in HIGHLIGHTED and USE_SELECTION:
            continue
        # frame1 = plot.plot_ellipsoid(frame1, X[it], Y[it], Z[it],
        #                              eX[it], eY[it], eZ[it])
        # frame1.set(xlabel='X', ylabel='Y', zlabel='Z')

        # frame2 = plot.plot_ellipsoid(frame2, U[it], V[it], W[it],
        #                              eU[it], eV[it], eW[it])
        # frame2.set(xlabel='U', ylabel='V', zlabel='W')

        s1 = plot.plot_ellipsoid_mayavi(fig1, X[it], Y[it], Z[it],
                                        eX[it], eY[it], eZ[it],
                                        color=colours[it][:3], rep='wireframe',
                                        name=names[it])
        #mlab.orientation_axes(xlabel='X', ylabel='Y', zlabel='Z', figure=fig1)

        s2 = plot.plot_ellipsoid_mayavi(fig2, U[it], V[it], W[it],
                                        eU[it], eV[it], eW[it],
                                        color=colours[it][:3], rep='wireframe',
                                        name=names[it])
        #mlab.orientation_axes(xlabel='U', ylabel='V', zlabel='W', figure=fig2)


    mlab.show()




# =============================================================================
# End of code
# =============================================================================
