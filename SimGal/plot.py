#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# CODE NAME HERE

# CODE DESCRIPTION HERE

Created on 2018-02-06 at 15:21

@author: cook



Version 0.0.0
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import sys
if sys.version_info.major == 2:
    from mayavi import mlab
else:
    mlab = None


# =============================================================================
# Define variables
# =============================================================================
COLOUR_RGB = dict(r=(1.0, 0.0, 0.0), y=(1.0, 1.0, 0.0))
# -----------------------------------------------------------------------------

# =============================================================================
# Define functions
# =============================================================================
def plot_ellipsoid(frame, xpos, ypos, zpos, xrad, yrad, zrad, color='k'):
    # Set of all spherical angles:
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    # Radii corresponding to the coefficients:
    rx, ry, rz = xrad, yrad, zrad
    # Cartesian coordinates that correspond to the spherical angles:
    # (this is the equation of an ellipsoid):
    x = rx * np.outer(np.cos(u), np.sin(v))
    y = ry * np.outer(np.sin(u), np.sin(v))
    z = rz * np.outer(np.ones_like(u), np.cos(v))
    # shift the center
    xi, yi, zi = x + xpos, y + ypos, z + zpos

    # Plot:
    frame.plot_wireframe(xi, yi, zi, rstride=4, cstride=4, color=color)

    return frame


def gen_rgb_colors(N):
    # Generate color map
    cm = plt.get_cmap('gist_rainbow')
    colors = []
    for i in range(1, N+1):
        color = cm(1. * i / N)  # color will now be an RGBA tuple
        colors.append(color)
    return colors

def plot_ellipsoid_mayavi(fig, xpos, ypos, zpos, xrad, yrad, zrad,
                          color=(1, 0, 0), rep='surface', name=''):
    # Set of all spherical angles:
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    # Radii corresponding to the coefficients:
    rx, ry, rz = xrad, yrad, zrad
    # Cartesian coordinates that correspond to the spherical angles:
    # (this is the equation of an ellipsoid):
    x = rx * np.outer(np.cos(u), np.sin(v))
    y = ry * np.outer(np.sin(u), np.sin(v))
    z = rz * np.outer(np.ones_like(u), np.cos(v))
    # shift the center
    xi, yi, zi = x + xpos, y + ypos, z + zpos

    # Plot:
    s = mlab.mesh(xi, yi, zi, color=color,
                  resolution=16, representation=rep,
                  line_width=0.5, figure=fig, name=name + '-e')
    s.actor.property.lighting = False

    # plot text

    t = mlab.text3d(xpos, ypos, zpos, text=name, figure=fig, color=color,
                    name=name + '-t')

    return s



def set_limits(frame, positions):

    # Adjustment of the axes, so that they all have the same span:
    max_radius = max(positions)
    for axis in 'xyz':
        getattr(frame, 'set_{}lim'.format(axis))((-max_radius, max_radius))

    return frame

# =============================================================================
# End of code
# =============================================================================
